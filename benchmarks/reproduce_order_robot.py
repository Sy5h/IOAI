"""Run/scoring harness for Find the Order and Robot Chasing.

Supply separately downloaded official data. Evaluation labels are passed only to
this parent scorer and never as an argument or environment variable to a solver.
This reproduces the metric, not the original Linux/Podman contest environment.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time


BASELINE_COMMIT = "b3e60ae8e14b55d97c44214dc1ec7d8e49295a89"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')


def prepare_holdout(source, destination, truth_dir, fraction=0.2, seed=20261016):
    source, destination, truth_dir = map(Path, (source, destination, truth_dir))
    observations = read_json(source / 'observations.json')
    labels = read_json(source / 'labels.json')
    groups = defaultdict(list)
    for i, row in enumerate(observations):
        groups[row['robot_id']].append(i)
    rng = random.Random(seed)
    train_indices, test_indices = [], []
    for robot in sorted(groups):
        indices = groups[robot]
        rng.shuffle(indices)
        count = int(len(indices) * fraction)
        test_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    write_json(destination / 'train' / 'observations.json', [observations[i] for i in train_indices])
    write_json(destination / 'train' / 'labels.json', [labels[i] for i in train_indices])
    write_json(destination / 'test_public' / 'observations.json', [observations[i] for i in test_indices])
    write_json(truth_dir / 'holdout_answers.json', [labels[i] for i in test_indices])
    manifest = {'source': str(source.resolve()), 'source_observations_sha256': sha256(source / 'observations.json'),
                'source_labels_sha256': sha256(source / 'labels.json'), 'seed': seed,
                'fraction': fraction, 'train_rows': len(train_indices), 'test_rows': len(test_indices),
                'train_indices': train_indices, 'test_indices': test_indices}
    write_json(truth_dir / 'holdout_manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith('indices')}))


def robot_metric(predictions, truth, observations):
    if len(predictions) != len(truth) or len(truth) != len(observations):
        raise ValueError('prediction, truth and observation lengths differ')
    if any(type(x) is not int or x not in range(6) for x in predictions):
        raise ValueError('actions must be integer 0..5')
    groups = defaultdict(list)
    for p, y, row in zip(predictions, truth, observations):
        groups[row['robot_id']].append(int(p == y))
    by_robot = {str(k): 100 * sum(v) / len(v) for k, v in sorted(groups.items())}
    return {'metric': 'mean_per_robot_accuracy_percent',
            'score': sum(by_robot.values()) / len(by_robot), 'by_robot': by_robot,
            'rows': len(truth), 'robots': len(groups)}


def order_metric(predictions, truth):
    values = []
    invalid = 0
    for dialogue, target in truth.items():
        pred = predictions.get(dialogue)
        n = len(target)
        if (not isinstance(pred, list) or len(pred) != n or
            any(type(x) is not int for x in pred) or sorted(pred) != list(range(n))):
            values.append(0.0)
            invalid += 1
            continue
        pairs = n * (n - 1) // 2
        correct = sum((pred[i] > pred[j]) == (target[i] > target[j])
                      for i in range(n) for j in range(i + 1, n))
        values.append(correct / pairs if pairs else 1.0)
    return {'metric': 'mean_dialogue_pairwise_order_accuracy_percent',
            'score': 100 * sum(values) / len(values), 'dialogues': len(values),
            'invalid_dialogues': invalid}


def environment(device, threads):
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for name in ('numpy', 'scipy', 'torch', 'transformers', 'scikit-learn', 'huggingface-hub'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                       '--format=csv,noheader'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    cpu_name = platform.processor()
    if os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                cpu_name = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return {'python': platform.python_version(), 'platform': platform.platform(), 'cpu_name': cpu_name,
            'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
            'gpu_inventory': gpu, 'selected_device': device, 'threads': threads, 'packages': packages}


def run(args):
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    script, dataset = Path(args.script).resolve(), Path(args.dataset).resolve()
    output = run_dir / ('answers.json' if args.task == 'order' else 'predictions.json')
    if output.exists():
        raise FileExistsError(f'Use a fresh run directory: {run_dir}')
    command = [sys.executable, '-u', str(script)]
    command += (['--test-dir', str(dataset / 'test_public')] if args.task == 'order'
                else ['--dataset-root', str(dataset)])
    command += ['--output', str(output)] + args.solver_args
    env = os.environ.copy()
    env['PYTHONFAULTHANDLER'] = '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = str(args.threads)
    if args.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    cache = run_dir.parent.parent / 'hf-cache'
    env['HF_HOME'] = str(cache)
    env['HF_HUB_CACHE'] = str(cache / 'hub')
    metadata = {'label': args.label, 'task': args.task, 'baseline_commit': BASELINE_COMMIT,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha256(script),
                'command': command, 'dataset': str(dataset), 'data_revision': args.data_revision,
                'evaluation_split': args.split, 'truth_sha256': sha256(args.truth),
                'environment': environment(args.device, args.threads),
                'timing_scope': 'solver subprocess wall clock; excludes downloads and scorer',
                'repeat_index': args.repeat_index, 'original_contest_environment': False}
    print(json.dumps({'label': args.label, 'command': command}), flush=True)
    started = time.perf_counter()
    with (run_dir / 'solver.log').open('w', encoding='utf-8') as log:
        try:
            result = subprocess.run(command, cwd=run_dir, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout)
            metadata['returncode'] = result.returncode
            metadata['status'] = 'completed' if result.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            metadata.update(status='timeout', returncode=None)
    metadata['run_seconds'] = time.perf_counter() - started
    text = (run_dir / 'solver.log').read_text(encoding='utf-8', errors='replace')
    metadata['solver_logged_fallback'] = any(s in text for s in
        ('training failed;', 'ASR stopped;', 'Qwen stopped;', 'training budget exhausted'))
    metadata['log_tail'] = text[-2000:]
    if metadata['status'] == 'completed' and output.is_file():
        # This is the only phase that reads evaluation labels.
        predictions, truth = read_json(output), read_json(args.truth)
        metadata['result'] = (order_metric(predictions, truth) if args.task == 'order'
                              else robot_metric(predictions, truth,
                                    read_json(dataset / 'test_public' / 'observations.json')))
    write_json(run_dir / 'result.json', metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    prepare = sub.add_parser('prepare-holdout')
    prepare.add_argument('--source', required=True)
    prepare.add_argument('--destination', required=True)
    prepare.add_argument('--truth-dir', required=True)
    prepare.add_argument('--fraction', type=float, default=0.2)
    prepare.add_argument('--seed', type=int, default=20261016)
    execute = sub.add_parser('run')
    execute.add_argument('--task', choices=['robot', 'order'], required=True)
    execute.add_argument('--label', required=True)
    execute.add_argument('--script', required=True)
    execute.add_argument('--dataset', required=True)
    execute.add_argument('--truth', required=True)
    execute.add_argument('--run-dir', required=True)
    execute.add_argument('--data-revision', required=True)
    execute.add_argument('--split', required=True)
    execute.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    execute.add_argument('--threads', type=int, default=4)
    execute.add_argument('--timeout', type=float, default=330)
    execute.add_argument('--repeat-index', type=int, default=1)
    execute.add_argument('solver_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == 'prepare-holdout':
        prepare_holdout(args.source, args.destination, args.truth_dir, args.fraction, args.seed)
    else:
        if args.solver_args and args.solver_args[0] == '--':
            args.solver_args = args.solver_args[1:]
        run(args)


if __name__ == '__main__':
    main()
