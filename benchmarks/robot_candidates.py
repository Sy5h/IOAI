import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('base', ROOT / 'solutions/day1_robot_chasing.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def compact(obs, raw_grid=True):
    engineered = base.feature_vector(obs)[1152:]
    raw = np.asarray(obs['image'], dtype=np.float32).reshape(-1)
    return np.concatenate((engineered, raw)) if raw_grid else engineered


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--budget', type=float, default=270)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--trees', type=int, default=128)
    p.add_argument('--max-features', type=float, default=0.8)
    p.add_argument('--min-leaf', type=int, default=2)
    p.add_argument('--model', choices=['extra-trees', 'hist'], default='extra-trees')
    p.add_argument('--no-raw-grid', action='store_true')
    a = p.parse_args()
    started = time.perf_counter()
    root = Path(a.dataset_root)
    train = json.loads((root/'train/observations.json').read_text())
    labels = json.loads((root/'train/labels.json').read_text())
    test = json.loads((root/'test_public/observations.json').read_text())
    defaults = base.majority_actions(train, labels)
    pred = [int(defaults[o['robot_id']]) for o in test]
    base.write_predictions(pred, a.output)
    for robot in range(6):
        indices = [i for i,o in enumerate(train) if o['robot_id']==robot]
        test_idx = [i for i,o in enumerate(test) if o['robot_id']==robot]
        if time.perf_counter()-started >= a.budget-15:
            break
        x = np.stack([compact(train[i], not a.no_raw_grid) for i in indices])
        y = np.asarray([labels[i] for i in indices])
        xt = np.stack([compact(test[i], not a.no_raw_grid) for i in test_idx])
        if a.model == 'hist':
            model = HistGradientBoostingClassifier(max_iter=a.trees, learning_rate=0.1,
                     max_leaf_nodes=31, l2_regularization=1, early_stopping=True,
                     validation_fraction=0.15,n_iter_no_change=15,random_state=2026+robot)
        else:
            model=ExtraTreesClassifier(n_estimators=a.trees, max_features=a.max_features,
                                      min_samples_leaf=a.min_leaf,n_jobs=a.cpu_threads,
                                      random_state=2026+robot)
        model.fit(x,y)
        yp=model.predict(xt)
        for i,action in zip(test_idx,yp):
            pred[i]=int(action)
        base.write_predictions(pred,a.output)
        print(f'robot={robot} elapsed={time.perf_counter()-started:.2f} features={x.shape[1]}',flush=True)
    print(f'rows={len(pred)} elapsed={time.perf_counter()-started:.3f}',flush=True)


if __name__=='__main__':
    main()
