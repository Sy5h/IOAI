"""Cross-check measured outputs with pinned official scoring functions.

This executes only the named metric/reporting function definitions, with local
input paths and a capture-only write_report. It does NOT run the official
Podman/GPU/timeout/notebook grading environment.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

OFFICIAL_REVISION = 'e1963801aa4cf33a5a03520b39fa6620a7894c54'
SOURCES = {
    'order': ('1_Find_the_Order', 'ad599419f5e827157a443266fbe88ea1274d1e9bf8652dc127beb3e772548dc9'),
    'robot': ('2_Robot_Chasing', '4a0a23a4cd521ee90b93b60eb3733b7d4d1d19dfd4fbea1971cf2e6b302395d8'),
}


def functions(path, task):
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != SOURCES[task][1]:
        raise ValueError(f'Official {task} source hash does not match pinned revision')
    names = ({'validate_permutation', 'score_one', 'score_submission'} if task == 'order'
             else {'_flatten_ints', 'score_submission'})
    tree = ast.parse(content.decode('utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('Missing expected official scoring functions')
    return compile(ast.Module(body=nodes, type_ignores=[]), '<pinned_official_scoring_functions>', 'exec')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-dir', type=Path, required=True,
                        help='Folder containing pinned check.py files renamed order.py and robot.py')
    parser.add_argument('--bench-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('scoring_audit_order_robot.json'))
    args = parser.parse_args()
    compiled = {task: functions(args.official_dir / (task + '.py'), task) for task in SOURCES}
    rows = []
    for result_path in sorted((args.bench_dir / 'runs').glob('*/result.json')):
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result['status'] != 'completed' or 'result' not in result:
            continue
        task, split = result['task'], result['evaluation_split']
        if split.startswith('train_holdout'):
            truth_name = 'holdout_answers.json'
        elif 'pretrain' in split:
            truth_name = 'pretrain_answers.json'
        elif 'pretest' in split:
            truth_name = 'pretest_answers.json'
        else:
            truth_name = split + '_answers.json'
        captured = []

        def fail(message, **kwargs):
            raise ValueError(message)

        namespace = {
            'json': json, 'sys': sys, 'NUM_ACTIONS': 6, 'NUM_ROBOTS': 6,
            'SUBMISSION_PATH': result_path.parent / ('answers.json' if task == 'order' else 'predictions.json'),
            'HIDDEN_ANSWERS': args.bench_dir / 'scorer-only' / task / truth_name,
            'HIDDEN_OBS': Path(result['dataset']) / 'test_public' / 'observations.json',
            'fail': fail, 'write_report': lambda **fields: captured.append(fields),
        }
        exec(compiled[task], namespace)
        namespace['score_submission']()
        official = captured[0]['score']
        local = round(result['result']['score'], 4)
        row = {'run_id': result_path.parent.name, 'official_score_rounded_4dp': official,
               'benchmark_score_rounded_4dp': local, 'matches': official == local,
               'official_details': captured[0]['details']}
        rows.append(row)
        if not row['matches']:
            raise AssertionError(row)
    report = {
        'official_source_revision': OFFICIAL_REVISION,
        'scope': 'Execute pinned original score_submission and its metric helper functions only; no Podman, GPU, timeout or notebook wrapper.',
        'sources': {task: {
            'url': f'https://github.com/IOAI-official/IOAI-2026/blob/{OFFICIAL_REVISION}/Individual-Contest/{directory}/code/grading-original/check.py',
            'sha256': digest} for task, (directory, digest) in SOURCES.items()},
        'all_match': all(row['matches'] for row in rows), 'runs': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Official metric cross-check: {len(rows)}/{len(rows)} runs match at 4 decimal places')


if __name__ == '__main__':
    main()
