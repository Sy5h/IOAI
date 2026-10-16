"""Choose simple parameters using TRAIN article holdout only, never test labels."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '4'
from pathlib import Path
import argparse
import importlib.util
import hashlib
import json
import time
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-root', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    data_root = Path(args.data_root).resolve()
    spec = importlib.util.spec_from_file_location('ghost', REPO/'solutions/day2_ghost_of_the_machine.py')
    ghost = importlib.util.module_from_spec(spec); spec.loader.exec_module(ghost)
    train_dir = data_root/'ghost-source/public/train'
    rows = ghost.read_jsonl(train_dir/'data.jsonl')
    answers = ghost.read_answers(train_dir)
    fit, val = ghost.article_split(rows, .15, 42)
    fitting = [rows[i] for i in fit]
    separators = ghost.learn_separators(fitting, answers)
    mean = np.mean([answers[r['id']]/len(r['text']) for r in fitting])
    sentences, y, article, spans = ghost.build_sentence_data(rows, separators, answers)
    cache = data_root/'ghost_train_features.npz'
    cache_key = hashlib.sha256(json.dumps({
        'texts': sentences, 'model_revision': 'a5beb1e3e68b9ab74eb54cfd186867f64f240e1a',
        'pooling': 'normalized CLS', 'max_length': 128, 'precision': 'float16 CUDA / float32 CPU',
        'device': args.device,
    }, ensure_ascii=False).encode()).hexdigest()
    import torch
    torch.set_num_threads(4)
    reused_cache = False
    cached = np.load(cache) if cache.exists() else None
    if cached is not None and 'cache_key' in cached and str(cached['cache_key']) == cache_key:
        x = cached['features']
        reused_cache = True
    else:
        t = time.monotonic()
        x = ghost.embed_sentences(sentences, data_root/'models/bge-base-en-v1.5', device=args.device)
        np.savez_compressed(cache, features=x, cache_key=cache_key)
        print(json.dumps({'embedding_seconds': time.monotonic()-t, 'sentences': len(sentences)}), flush=True)
    from sklearn.linear_model import LogisticRegression
    mask = np.isin(article, fit)&(y>=0)
    val_answers = {rows[i]['id']: answers[rows[i]['id']] for i in val}
    records = []
    for c in [2.0, 10.0, 100.0, 1000.0]:
        t = time.monotonic()
        model = LogisticRegression(C=c, max_iter=150, solver='lbfgs', random_state=42)
        model.fit(x[mask], y[mask])
        probabilities = model.predict_proba(x)[:,1]
        for prior in [0.0, .25, 1.0]:
            for expected in [False, True]:
                options = SimpleNamespace(prior_strength=prior, min_side_words=180, expected_score=expected)
                predictions = ghost.decode_articles(rows, spans, article, probabilities, mean, options)
                score = ghost.boundary_score(predictions, val_answers)
                row={'C': c, 'prior_strength': prior, 'expected_score': expected, 'score': score,
                     'validation_articles': len(val), 'fit_articles': len(fit), 'elapsed':time.monotonic()-t}
                records.append(row); print(json.dumps(row), flush=True)
    best = max(records, key=lambda x: x['score'])
    result = {'split':'TRAIN article holdout; no test rows or labels accessed', 'seed':42,
              'separators':separators, 'mean_fraction':mean, 'records':records, 'selected':best,
              'run_seconds':time.monotonic()-started, 'threads':4, 'device':args.device,
              'cache_reused': reused_cache, 'cache_key': cache_key,
              'length_bucketing': False,
              'timing_scope': 'development-only tuning; not a final solver runtime'}
    (data_root/'results').mkdir(exist_ok=True)
    (data_root/'results/ghost_tuning.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('SELECTED',json.dumps(best),flush=True)

if __name__=='__main__':
    main()
