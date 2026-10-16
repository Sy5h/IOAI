"""Run a Ghost solution, then score in this separate parent process.

No test answers are passed to the solution. Download/model preparation is outside
the timed region. This is a local reproduction, not a contest-server submission.
"""
import argparse
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

BASELINE_COMMIT = 'b3e60ae8e14b55d97c44214dc1ec7d8e49295a89'
DATA_REVISION = '6dcd17d529f154935cda550c6e80f9801887a316'
MODEL_REVISION = 'a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def score_predictions(path, test_path, answers_path):
    rows, predictions, labels = read_jsonl(test_path), read_jsonl(path), read_jsonl(answers_path)
    ids = [r['id'] for r in rows]
    prediction_ids = [r['id'] for r in predictions]
    if len(set(prediction_ids)) != len(prediction_ids) or set(ids) != set(prediction_ids):
        raise ValueError('Missing, duplicate, or unknown prediction ID')
    text_lengths = {r['id']:len(r['text']) for r in rows}
    answers = {r['id']:r['boundary_char_index'] for r in labels}
    if set(answers) != set(ids):
        raise ValueError('Evaluation answer IDs do not match test IDs')
    for r in predictions:
        value = r['boundary_char_index']
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= text_lengths[r['id']]:
            raise ValueError('Boundary must be an integer inside the original text')
    errors = [abs(r['boundary_char_index']-answers[r['id']]) for r in predictions]
    return {'score':100*sum(math.exp(-e/100) for e in errors)/len(errors),
            'mean_absolute_character_error':sum(errors)/len(errors),
            'exact_boundaries':sum(e==0 for e in errors), 'count':len(errors)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', default=str(Path(__file__).resolve().parents[1]/'solutions/day2_ghost_of_the_machine.py'))
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--answers', required=True, help='SCORER ONLY; never passed to the child solver')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--time-limit', type=float, default=600)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--solver-args', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    prediction_path = output/(args.tag+'.predictions.jsonl')
    script = Path(args.script).resolve()
    command = [sys.executable, '-u', str(script), '--train-dir',str(Path(args.train_dir).resolve()),
               '--test-dir',str(Path(args.test_dir).resolve()), '--model-dir',str(Path(args.model_dir).resolve()),
               '--output',str(prediction_path), *args.solver_args]
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key] = str(args.threads)
    env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
    log_path = output/(args.tag+'.log')
    result = {'task':'ghost', 'split':args.split, 'baseline_commit':BASELINE_COMMIT,
              'data_revision':DATA_REVISION, 'model_revision':MODEL_REVISION,
              'metric':'100 * mean(exp(-abs(predicted_character - true_character)/100))',
              'repetitions':1,'download_included':False, 'command':command,
              'parameters': {'solver_args': args.solver_args, 'whole_run_limit': args.time_limit},
              'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
              'hardware':{'platform':platform.platform(),'cpu':platform.processor(),
                          'logical_cpus':os.cpu_count(),'threads':args.threads,'python':sys.version},
              'local_reproduction':True, 'sandbox':'separate processes; not an OS security sandbox'}
    result['hardware']['libraries']={}
    for package in ['numpy','torch','transformers','scikit-learn','safetensors']:
        try:
            result['hardware']['libraries'][package]=version(package)
        except PackageNotFoundError:
            result['hardware']['libraries'][package]='not installed'
    try:
        result['hardware']['gpu'] = subprocess.check_output(
            ['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        result['hardware']['gpu'] = 'not detected'
    started=time.monotonic()
    try:
        with log_path.open('w',encoding='utf-8') as log:
            completed=subprocess.run(command,cwd=script.parent,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     timeout=args.time_limit,check=False)
        result['run_seconds']=time.monotonic()-started
        result['returncode']=completed.returncode
        if completed.returncode:
            raise RuntimeError(f'Solver exit code {completed.returncode}; see log')
        result.update(score_predictions(prediction_path,Path(args.test_dir)/'data.jsonl',args.answers))
        result['status']='fallback' if 'kept legal-boundary fallback' in log_path.read_text(encoding='utf-8') else 'ok'
    except Exception as error:
        result['run_seconds']=time.monotonic()-started
        result['status']='failed';result['error']=repr(error)
    (output/(args.tag+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if result['status']=='failed':
        sys.exit(1)

if __name__=='__main__':
    main()
