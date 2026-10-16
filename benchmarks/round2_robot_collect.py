"""Collect Robot round-2 measurements and independently apply the pinned scorer."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--official-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('round2_robot_results.json'))
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    collect = load('round2_sanitize', directory / 'collect_order_robot.py')
    audit = load('round2_official', directory / 'audit_order_robot_scores.py')
    compiled = audit.functions(args.official_dir / 'robot.py', 'robot')
    logs = args.output.parent / 'logs' / 'round2_robot'
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.runs.glob('*/result.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        row['run_id'] = path.parent.name
        row['repeated_measurements'] = False
        row['training_rows'] = 48000 if row['evaluation_split'].startswith('train_holdout') else 60000
        snapshot = {'paths_hist_holdout': 'candidate_v1', 'paths_compact_mlp_holdout': 'candidate_v1',
                    'relative_grid_mlp_holdout': 'candidate_v2'}.get(row['run_id'])
        if snapshot:
            manifest = args.runs.parent / snapshot / 'sha256.json'
            if manifest.exists():
                row['candidate_source_snapshot_sha256'] = json.loads(manifest.read_text())
                row['candidate_source_note'] = 'Source snapshot preserved during development. The current reproduction wrapper adds variants/compatibility but retains this variant feature calculation.'
        row['parameters'] = {'seed': 2026, 'budget_seconds': 270, 'threads': 4,
                             'internal_validation_fraction': 0.15}
        if isinstance(row.get('dependency_source_sha256'), dict):
            row['dependency_source_sha256'] = [{'path': key, 'sha256': value}
                                                for key, value in row['dependency_source_sha256'].items()]
        if 'hist' in row['label']:
            row['parameters'].update(model='HistGradientBoostingClassifier', features=404,
                                      max_iter=128, max_leaf_nodes=31, learning_rate=0.1,
                                      l2_regularization=1, patience=15)
        else:
            row['parameters'].update(model='MLP', hidden=[192, 64], max_epochs=60,
                                      learning_rate=0.002, batch_size=512, dropout=0.05,
                                      weight_decay=0.0001, patience=8)
            if 'baseline' in row['label']:
                row['parameters']['features'] = 1328
            elif 'local' in row['label']:
                row['parameters']['features'] = 1622
            elif 'relative' in row['label']:
                row['parameters']['features'] = 1866
            elif 'target' in row['label']:
                row['parameters']['features'] = 1518
            elif 'compact' in row['label']:
                row['parameters']['features'] = 404
        log = (path.parent / 'solver.log').read_text(encoding='utf-8', errors='replace')
        learned = {int(match.group(1)) for match in re.finditer(
            r'^robot (\d+): .*?(?:learned=True|hist iterations=\d+)', log, re.MULTILINE)}
        row['training_coverage'] = {'learned_robot_ids': sorted(learned), 'learned_robots': len(learned),
                                    'total_robots': 6, 'fallback_robots': 6 - len(learned)}
        target = logs / (path.parent.name + '.log')
        target.write_text(collect.sanitize(log, args.workspace.resolve()), encoding='utf-8')
        row['sanitized_solver_log'] = 'benchmarks/logs/round2_robot/' + target.name
        if row['status'] == 'completed' and 'result' in row:
            captured = []

            def fail(message, **kwargs):
                raise ValueError(message)

            namespace = {'json': json, 'sys': sys, 'NUM_ACTIONS': 6, 'NUM_ROBOTS': 6,
                         'SUBMISSION_PATH': path.parent / 'predictions.json',
                         'HIDDEN_ANSWERS': Path(row['truth_source']),
                         'HIDDEN_OBS': Path(row['dataset']) / 'test_public' / 'observations.json',
                         'fail': fail, 'write_report': lambda **fields: captured.append(fields)}
            exec(compiled, namespace)
            namespace['score_submission']()
            official = captured[0]['score']
            row['official_score_audit'] = {'score_rounded_4dp': official,
                                            'matches': official == round(row['result']['score'], 4)}
            if not row['official_score_audit']['matches']:
                raise AssertionError(row['run_id'])
        rows.append(collect.sanitize(row, args.workspace.resolve()))
    report = {
        'baseline_commit': '991a49b48092286967b5bd7cada465e336ebc4af',
        'data_revision': '8a82acbdfda387269649d4bcf580fbdab1ed3cc6',
        'evaluation_status': 'Development re-evaluation: released A/B were already measured in round 1. They are not new unseen test sets.',
        'selection_protocol': 'Use only official train, same robot-stratified 48000/12000 split (seed 20261016), to select the candidate. Freeze before A/B re-evaluation. No fitting or tuning on evaluation answers.',
        'timing_scope': 'Fresh CPU solver subprocess, 4 threads, includes imports, JSON loading, features, training and prediction; excludes downloads and scorer. Shared Windows host; concurrent Order inference and other work can affect CPU wall time. Each run once; timing differences are not controlled dedicated-host speed estimates.',
        'downloads': {'performed_in_round_2': False, 'note': 'Reuse previously downloaded pinned data; no model or additional data download.'},
        'official_score_audit': {'revision': audit.OFFICIAL_REVISION,
                                  'source_sha256': audit.SOURCES['robot'][1],
                                  'scope': 'Original scoring function only, not the full official Podman/notebook/time-limit environment.'},
        'runs': rows,
    }
    selection = args.runs.parent / 'selection.json'
    if selection.exists():
        report['selection_frozen_before_A_B'] = json.loads(selection.read_text(encoding='utf-8'))
    indexed = {row['run_id']: row for row in rows}
    comparisons = []
    for split in ('a', 'b'):
        old, new = indexed.get('baseline_' + split), indexed.get('selected_local_' + split)
        if old and new and 'result' in old and 'result' in new:
            comparisons.append({'split': 'test_leaderboard_' + split,
                                'baseline_score': old['result']['score'],
                                'selected_score': new['result']['score'],
                                'gain_percentage_points': new['result']['score'] - old['result']['score'],
                                'baseline_seconds': old['run_seconds'],
                                'selected_seconds': new['run_seconds']})
    report['paired_development_re_evaluation'] = comparisons
    if len(comparisons) == 2 and all(row['gain_percentage_points'] > 0 for row in comparisons):
        report['release_review_after_A_B'] = 'Retain the preselected local MLP feature default after gains on both released A/B sets. No hyperparameter or feature tuning followed A/B. Timing remains a single shared-host observation.'
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Collected {len(rows)} runs; completed official scores match')


if __name__ == '__main__':
    main()
