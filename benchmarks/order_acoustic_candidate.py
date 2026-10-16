"""Development-only ablation: fixed prefix + conservative speaker alternation.

No ASR, language model, training labels or evaluation labels are read. This is
an exploratory candidate, not the main solution or a claimed improvement.
"""
import argparse
import importlib.util
from pathlib import Path
import time

spec = importlib.util.spec_from_file_location(
    'order_solution', Path(__file__).resolve().parents[1] / 'solutions/day1_find_the_order.py')
order = importlib.util.module_from_spec(spec)
spec.loader.exec_module(order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    records = order.load_dialogues(args.test_dir)
    accepted = 0
    for record in records:
        features = [order.acoustic_features(order.read_audio(path)) for path in record['paths']]
        speakers = order.reliable_speakers(features, record['prefix'])
        if speakers is not None:
            accepted += 1
            edges = [[0.0] * len(features) for _ in features]
            record['order'] = order.beam_order(edges, record['prefix'], speakers=speakers)
    order.write_answers(records, args.output)
    print(f'speaker constraints accepted for {accepted}/{len(records)} dialogues; elapsed={time.perf_counter()-started:.3f}s')


if __name__ == '__main__':
    main()
