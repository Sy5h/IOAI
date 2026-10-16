"""Fresh-process CPU measurements for Robot round 2; evaluation is development re-evaluation."""
import argparse
import importlib.util
import json
from pathlib import Path

BASELINE_COMMIT = "991a49b48092286967b5bd7cada465e336ebc4af"
DATA_REVISION = "8a82acbdfda387269649d4bcf580fbdab1ed3cc6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--script', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--timeout', type=float, default=330)
    parser.add_argument('solver_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.solver_args[:1] == ['--']:
        args.solver_args = args.solver_args[1:]
    spec = importlib.util.spec_from_file_location('round2_harness', Path(__file__).with_name('reproduce_order_robot.py'))
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.BASELINE_COMMIT = BASELINE_COMMIT
    args.task, args.device, args.threads = 'robot', 'cpu', 4
    args.data_revision, args.repeat_index = DATA_REVISION, 1
    dependencies = {}
    if Path(args.script).name == 'round2_robot_candidate.py':
        feature_source = Path(args.script).with_name('round2_robot_features.py')
        dependencies[str(feature_source.resolve())] = harness.sha256(feature_source)
        if '--source' in args.solver_args:
            baseline = Path(args.solver_args[args.solver_args.index('--source') + 1])
            dependencies[str(baseline.resolve())] = harness.sha256(baseline)
    result = harness.run(args)
    result['round'] = 2
    result['dependency_source_sha256'] = dependencies
    result['truth_source'] = str(Path(args.truth).resolve())
    result['evaluation_role'] = ('training_only_holdout' if args.split.startswith('train_holdout')
                                 else 'released_A_B_development_re_evaluation')
    result['selection_protocol'] = 'Select on the same 48000/12000 training-only split; freeze before released A/B. A/B were evaluated in round 1 and are not fresh unseen test sets.'
    harness.write_json(Path(args.run_dir) / 'result.json', result)


if __name__ == '__main__':
    main()
