"""Local reproduction of the original judge's interactive protocol.

The solver runs as a child process. Secrets are read only here, never passed in
arguments, environment, or input JSON; only official comparison feedback is sent.
Public/dev simulations use public embeddings. Released A/B evaluations use
scorer-only private embeddings. Neither is a contest-server submission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import subprocess
import sys
import threading
import time

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '4'
import numpy as np

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
BASELINE_COMMIT = 'b3e60ae8e14b55d97c44214dc1ec7d8e49295a89'
DATA_REVISION = '122716fc56a212e58b1756158bdeb4c2ea2da5e6'

def readline_worker(stream, lines):
    try:
        for line in stream:
            lines.put(line)
    finally:
        lines.put(None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', default=str(REPOSITORY / 'solutions/day1_potato.py'))
    parser.add_argument('--data-dir', required=True, help='Public vocabulary, embeddings, and test_public.json')
    parser.add_argument('--suite', choices=['public', 'dev', 'leaderboard_a', 'leaderboard_b'], default='public')
    parser.add_argument('--evaluation-dir', help='Scorer-only released private embedding/target directory')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--count', type=int, default=120)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--tag', default='current-public')
    parser.add_argument('--solver-args', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    data = Path(args.data_dir)
    words = json.loads((data / 'vocabulary.json').read_text(encoding='utf-8'))
    secrets = json.loads((data / 'test_public.json').read_text(encoding='utf-8'))
    if isinstance(secrets, dict):
        secrets = secrets.get('words', secrets.get('secrets', list(secrets)))
    if args.suite == 'dev':
        pool = sorted(set(words) - set(secrets))
        secrets = np.random.default_rng(args.seed).choice(pool, min(args.count, len(pool)), replace=False).tolist()
    elif args.suite.startswith('leaderboard_'):
        secrets = json.loads((Path(args.evaluation_dir) / ('secrets_' + args.suite + '.json')).read_text(encoding='utf-8'))[:args.count]
    else:
        secrets = secrets[:args.count]
    index = {word.casefold(): i for i, word in enumerate(words)}
    embedding_path = (Path(args.evaluation_dir) / 'private_embeddings.npy'
                      if args.suite.startswith('leaderboard_') else data / 'public_embeddings.npy')
    embeddings = np.load(embedding_path).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / (args.tag + '.stderr.log')
    script = Path(args.script).resolve()
    env = os.environ.copy()
    env['POTATO_DATA_DIR'] = str(data.resolve())
    command = [sys.executable, '-u', str(script), *args.solver_args]
    records = []
    failure = None
    started = time.monotonic()
    with log_path.open('w', encoding='utf-8') as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=stderr, text=True, encoding='utf-8', env=env,
                                   cwd=script.parent)
        lines = queue.Queue()
        reader = threading.Thread(target=readline_worker, args=(process.stdout, lines), daemon=True)
        reader.start()
        def send(message):
            process.stdin.write(json.dumps(message) + '\n')
            process.stdin.flush()
        try:
            for game, secret in enumerate(secrets):
                send({'event': 'new_game'})
                first, second = 'lamp', 'potato'
                found = None
                secret_vector = embeddings[index[secret.casefold()]]
                game_started = time.monotonic()
                for turn in range(1, 31):
                    difference = float(secret_vector @ embeddings[index[first]]) - float(secret_vector @ embeddings[index[second]])
                    verdict = 'same' if abs(difference) <= 1e-12 else ('first' if difference > 0 else 'second')
                    winner = second if verdict == 'second' else first
                    send({'turn': turn, 'winner_word': winner, 'verdict': verdict,
                          'word1': first, 'word2': second})
                    remaining = args.seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError('Whole-suite time limit exceeded')
                    try:
                        line = lines.get(timeout=remaining)
                    except queue.Empty:
                        raise TimeoutError('No solver reply before suite deadline')
                    if line is None:
                        raise RuntimeError('Solver exited before replying; see stderr log')
                    proposal = json.loads(line)['new_word'].casefold()
                    if proposal not in index:
                        raise ValueError('Proposal is outside the vocabulary')
                    if proposal == secret.casefold():
                        found = turn
                        send({'status': 'win'})
                        break
                    first, second = winner, proposal
                if found is None:
                    send({'status': 'loss'})
                records.append({'game': game, 'secret_sha256': hashlib.sha256(secret.encode()).hexdigest(),
                                'turn': found, 'score': 0 if found is None else 1 - .02 * max(0, found - 10),
                                'game_seconds': time.monotonic() - game_started})
                if (game + 1) % 20 == 0:
                    print(json.dumps({'completed': game + 1, 'total': len(secrets),
                                      'elapsed': time.monotonic() - started}), flush=True)
            send({'event': 'done'})
            process.stdin.close()
            returncode = process.wait(timeout=10)
            if returncode:
                raise RuntimeError(f'Solver exit code {returncode}')
        except Exception as error:
            failure = repr(error)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    seconds = time.monotonic() - started
    result = {'label': ('Local re-evaluation of released official private embeddings and targets; not a contest server run'
                        if args.suite.startswith('leaderboard_') else 'Local simulation on official PUBLIC embeddings; not official hidden score'),
              'suite': args.suite, 'games': len(records), 'wins': sum(r['turn'] is not None for r in records),
              'score': (100 * sum(r['score'] for r in records) / len(records)) if not failure else None,
              'run_seconds': seconds, 'command': command,
              'status': 'failed' if failure else 'ok', 'error': failure,
              'baseline_commit': BASELINE_COMMIT, 'data_revision': DATA_REVISION,
              'metric': '100 * mean(0 if not solved, else 1 - .02 * max(0, winning_turn - 10))',
              'repetitions': 1, 'download_included': False,
              'parameters': {'solver_args': args.solver_args, 'whole_run_limit': args.seconds},
              'hardware': {'platform': platform.platform(), 'cpu': platform.processor(),
                           'logical_cpus': os.cpu_count(), 'numpy': np.__version__, 'blas_threads': 4},
              'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
              'seed': args.seed, 'records': records}
    output = result_dir / (args.tag + '.json')
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2), flush=True)
    if failure:
        sys.exit(1)

if __name__ == '__main__':
    main()
