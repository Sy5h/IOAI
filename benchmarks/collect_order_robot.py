"""Collect measured run records and sanitized logs; never read answer files."""
import argparse
import json
from pathlib import Path
import re


def sanitize(value, workspace):
    if isinstance(value, dict):
        return {key: sanitize(item, workspace) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, workspace) for item in value]
    if not isinstance(value, str):
        return value
    value = re.sub(r'[A-Za-z]:[\\/]Users[\\/][^\\/\s]+[\\/]AppData[\\/]Local[\\/]Programs[\\/]Python[\\/][^\\/\s]+[\\/]python.exe', 'python', value)
    for root in (str(workspace), workspace.as_posix()):
        value = value.replace(root, '<workspace>')
    value = re.sub(r'[A-Za-z]:[\\/]Users[\\/][^\\/\s]+', '<user-home>', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--bench-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    runs = []
    log_dir = args.output_dir / 'logs' / 'order_robot'
    log_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((args.bench_dir / 'runs').glob('*/result.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        row['run_id'] = path.parent.name
        row['repeated_measurements'] = False
        if row['task'] == 'robot':
            row['training_rows'] = 48000 if row['evaluation_split'].startswith('train_holdout') else 60000
            if row['label'].startswith('compact_'):
                row['reproduction_command'] = list(row['command'])
                row['reproduction_command'][2] = str(Path(__file__).with_name('robot_candidates.py').resolve())
                row['reproduction_note'] = 'Portable copy of the development candidate; import path is adapted to this repository. Feature extraction and model parameters are unchanged.'
            row['parameters'] = {'budget_seconds': 270, 'seed': 2026, 'threads': 4}
            if 'hist' in row['label']:
                row['parameters'].update(model='HistGradientBoostingClassifier', features=176,
                                         max_iter=128, max_leaf_nodes=31, learning_rate=0.1,
                                         l2_regularization=1, validation_fraction=0.15, patience=15)
                if 'selected_hist' in row['label']:
                    row['reproduction_command'] = row['command'] + ['--model', 'hist']
                    row['reproduction_note'] = 'Measured when hist was the default. The final release restores the original mlp default; reproduce this measured HGB setting with explicit --model hist.'
            elif 'extratrees' in row['label']:
                row['parameters'].update(model='ExtraTreesClassifier', features=304,
                                         n_estimators=128, max_features=0.8, min_samples_leaf=2)
            else:
                row['parameters'].update(model='MLP', features=1328, hidden=[192, 64],
                                         max_epochs=60, learning_rate=0.002, batch_size=512,
                                         dropout=0.05, weight_decay=0.0001,
                                         validation_fraction=0.15, patience=8)
        else:
            row['parameters'] = {'prefix_only': '--prefix-only' in row['command'],
                                 'acoustic_only': row['label'].startswith('acoustic_only'),
                                 'solver_flags': row['command'][7:]}
        log = path.parent / 'solver.log'
        if log.exists():
            raw_log = log.read_text(encoding='utf-8', errors='replace')
            if row['task'] == 'order' and not (row['parameters']['prefix_only'] or row['parameters']['acoustic_only']):
                completed = sum(line.startswith('ordered dialogue ') for line in raw_log.splitlines())
                total = row.get('result', {}).get('dialogues')
                row['pipeline_coverage'] = {'qwen_ordered_dialogues': completed, 'total_dialogues': total,
                                            'prefix_fallback_dialogues': total - completed if total is not None else None}
                defaults = {'budget': 540, 'asr-fraction': 0.55, 'asr-batch': 8, 'lm-batch': 4,
                            'asr-tokens': 128, 'max-audio-seconds': 30, 'context-tokens': 96,
                            'target-tokens': 64, 'beam': 16}
                for key, default in defaults.items():
                    flag = '--' + key
                    value = row['command'][row['command'].index(flag) + 1] if flag in row['command'] else default
                    row['parameters'][key.replace('-', '_')] = type(default)(value)
                row['parameters']['speaker_constraints_enabled'] = '--no-speakers' not in row['command']
            target = log_dir / (path.parent.name + '.log')
            target.write_text(sanitize(raw_log, args.workspace.resolve()), encoding='utf-8')
            row['sanitized_solver_log'] = 'benchmarks/logs/order_robot/' + target.name
        runs.append(sanitize(row, args.workspace.resolve()))
    report = {
        'schema_version': 1,
        'official_metric_cross_check': 'benchmarks/scoring_audit_order_robot.json',
        'baseline_commit': 'b3e60ae8e14b55d97c44214dc1ec7d8e49295a89',
        'official_contest_environment': False,
        'timing_scope': 'Fresh solver subprocess wall time including imports, data loading, training and inference. Downloads and scorer excluded. Shared local Windows host; each configuration measured once.',
        'selection_protocol': {
            'robot': 'Split only official train by robot: seed 20261016, 48000 training / 12000 independent holdout. Select HGB on holdout before evaluating leaderboard A/B. No fitting or hyperparameter tuning used A/B labels. After the mixed final result, the release conservatively retained the original MLP default; see robot_release_decision.',
            'order': 'Use public/pretrain only for development. Public/pretest and leaderboard A/B answers remain scorer-only.',
        },
        'split_mapping': {
            'public/pretest': 'A public-test slice, not the full original public test: Robot 600 rows, Order 20 dialogues.',
            'private/test_leaderboard_a': 'Now-public post-contest leaderboard A evaluation.',
            'private/test_leaderboard_b': 'Now-public post-contest leaderboard B evaluation.',
        },
        'data_revisions': {
            'IOAI-official/ioai-2026-robot-chasing': '8a82acbdfda387269649d4bcf580fbdab1ed3cc6',
            'IOAI-official/ioai-2026-find-the-order': '71a35b5a566665e242416eba45f018dc378663b8',
        },
        'model_revisions': {
            'openai/whisper-small': '973afd24965f72e36ca33b3055d56a652f456b4d',
            'Qwen/Qwen2.5-0.5B': '060db6499f32faf8b98477b0a26969ef7d8b9987',
        },
        'downloads': {
            'download_seconds': None,
            'note': 'Download wall time was not measured separately. It is excluded from every solver measurement.',
            'recovered_failure': 'Subagent network DNS permission failure; parent downloaded identical pinned revisions under its network grant.',
            'additional_recovered_failure': 'First parallel Order A/B download failed with Windows PermissionError during parent-directory creation. Retried after serially creating all file and metadata directories; pinned files were reused.',
        },
        'input_integrity_checks': (json.loads((args.bench_dir / 'audio_integrity.json').read_text())
                                   if (args.bench_dir / 'audio_integrity.json').exists() else []),
        'setup_failures': ['First Order prefix-only B dispatch occurred before the data-layout copy finished and failed before solver launch because the scorer answer file was not yet present. No score or solver runtime was produced for that dispatch; retry used the completed layout.'],
        'robot_observation': 'HGB selected on train holdout gains 1.775 percentage points there, but final A loses 1.1667 points and final B gains 0.5833 points versus MLP. No consistent final-evaluation improvement is established. No further tuning or ensembling followed A/B evaluation.',
        'robot_release_decision': 'Keep the original MLP default because the proposed replacement did not demonstrate consistent independent improvement. HGB remains an explicit --model hist option. This conservative release decision was made after observing the mixed final results; those results are not claimed as an untouched validation of this release decision.',
        'post_measurement_fixes': ['Robot exception path preserves already-saved partial predictions. All listed completed Robot runs finished without that exception path.'],
        'order_development_diagnostic': {'split': 'public/pretrain', 'dialogues': 20,
                                         'speaker_constraints_accepted_dialogues': 15,
                                         'accepted_dialogues_with_all_labels_correct': 15,
                                         'accepted_clips': 157, 'correct_speaker_labels': 157,
                                         'note': 'Development-only speaker clustering diagnostic, not the contest ordering metric.'},
        'runs': runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'results_order_robot.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Collected {len(runs)} actual runs')


if __name__ == '__main__':
    main()
