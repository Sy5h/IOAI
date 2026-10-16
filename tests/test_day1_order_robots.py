"""Synthetic contract tests; these do not measure contest accuracy."""

import importlib.util
import itertools
import json
import math
from pathlib import Path
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "solutions" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


order = load_module("day1_find_the_order")
robots = load_module("day1_robot_chasing")


def observation(robot=0, mission="approach the red ball", direction=0):
    grid = [[[2, 0] if r in (0, 7) or c in (0, 7) else [1, 0]
             for c in range(8)] for r in range(8)]
    grid[2][2] = [10, 0]
    grid[1][2] = [6, 0]
    grid[4][5] = [5, 2]
    return {"robot_id": robot, "image": grid, "direction": direction,
            "mission": mission, "carrying": None}


class OrderTests(unittest.TestCase):
    def test_inverse_rank_and_invalid_order(self):
        self.assertEqual(order.order_to_ranks([2, 0, 1]), [1, 2, 0])
        with self.assertRaises(ValueError):
            order.order_to_ranks([0, 0, 2])

    def test_prefix_baseline_and_expected_pairwise_score(self):
        self.assertEqual(order.prefix_order(5, [3, 1]), [3, 1, 0, 2, 4])
        n = 7
        scores = []
        for suffix in itertools.permutations(range(2, n)):
            ranks = order.order_to_ranks([0, 1, *suffix])
            correct = sum(ranks[i] < ranks[j] for i in range(n) for j in range(i + 1, n))
            scores.append(correct / (n * (n - 1) / 2))
        expected = 1 - (n - 2) * (n - 3) / (2 * n * (n - 1))
        self.assertAlmostEqual(sum(scores) / len(scores), expected)

    def test_beam_prefix_and_uniqueness(self):
        edges = [[0.0] * 4 for _ in range(4)]
        edges[0][3] = 10
        edges[3][1] = 10
        result = order.beam_order(edges, [2, 0], width=2)
        self.assertEqual(result, [2, 0, 3, 1])
        self.assertEqual(sorted(result), [0, 1, 2, 3])

    def test_speaker_constraint_and_invalid_constraint_fallback(self):
        edges = [[0.0] * 4 for _ in range(4)]
        edges[1][3] = 10
        self.assertEqual(order.beam_order(edges, [0, 1], speakers=[0, 1, 0, 1]), [0, 1, 2, 3])
        self.assertEqual(order.beam_order(edges, [0, 1], speakers=[0, 0, 1, 1]), [0, 1, 3, 2])

    def test_prefix_only_cli_without_test_answers_or_audio_decoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "test" / "17"
            folder.mkdir(parents=True)
            for i in range(3):
                (folder / f"chunk_{i}.wav").touch()
            (root / "test" / "prefix.json").write_text(json.dumps({"17": [2, 0]}))
            output = root / "answers.json"
            order.main(["--test-dir", str(root / "test"), "--output", str(output), "--prefix-only"])
            self.assertEqual(json.loads(output.read_text()), {"17": [1, 2, 0]})

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
    def test_speaker_reliability(self):
        features = [[0, 0], [10, 10], [0.1, -0.1], [9.9, 10.1], [-0.1, 0.1]]
        self.assertEqual(order.reliable_speakers(features, [0, 1]), [0, 1, 0, 1, 0])
        self.assertIsNone(order.reliable_speakers([[0, 0]] * 5, [0, 1]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_qwen_target_mask_excludes_context_and_padding(self):
        import torch
        from types import SimpleNamespace

        class Tokenizer:
            pad_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [int(x) for x in text.split()]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, attention_mask, use_cache):
                logits = torch.zeros((*input_ids.shape, 5))
                logits[:, :, 2] = torch.arange(input_ids.shape[1])
                return SimpleNamespace(logits=logits)

        logp = lambda position: position - math.log(math.exp(position) + 4)
        model, tokenizer = Model(), Tokenizer()
        values = order.conditional_scores(model, tokenizer, [("1 1 1", "2 2"), ("1", "2")],
                                           2, 20, 20, time.monotonic() + 10)
        self.assertAlmostEqual(values[0], (logp(2) + logp(3)) / 2, places=6)
        self.assertAlmostEqual(values[1], logp(0), places=6)
        self.assertIsNone(order.conditional_scores(model, tokenizer, [("1", "2")],
                                                   1, 20, 20, time.monotonic() - 1))


class RobotTests(unittest.TestCase):
    def test_parser_preserves_source_destination_order(self):
        parsed = robots.parse_mission("Place the red box beside the blue key.")
        self.assertEqual(parsed["family"], 2)
        self.assertEqual(parsed["pairs"], [(7, 0), (5, 2)])
        self.assertEqual(robots.parse_mission("pick the grey ball up")["family"], 1)
        self.assertEqual(robots.parse_mission("get close to the green ball")["family"], 0)
        self.assertEqual(robots.parse_mission("get the green ball")["family"], 1)
        self.assertEqual(robots.parse_mission("grab the gray-colored key")["pairs"], [(5, 5)])

    def test_absolute_facing_and_blocked_orientation(self):
        item = observation(direction=3)
        self.assertEqual(robots.front_cell(item), ((2, 3), (1, 0)))
        move = robots.movement_observation(item, 0)
        self.assertEqual(move["attempted_cell"], (1, 2))
        self.assertEqual(move["object"], 6)
        self.assertEqual(move["new_direction"], 0)
        self.assertEqual(item["direction"], 3)  # a feature extractor does not mutate input

    def test_per_robot_majority(self):
        items = [observation(0), observation(0), observation(1), observation(1), observation(1)]
        defaults = robots.majority_actions(items, [2, 2, 3, 3, 4])
        self.assertEqual(defaults[0], 2)
        self.assertEqual(defaults[1], 3)
        with self.assertRaises(ValueError):
            robots.majority_actions(items, [0])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
    def test_features_keep_distractor_token_and_carrying(self):
        import numpy as np

        original = observation(mission="put the red ball next to the blue key")
        changed = json.loads(json.dumps(original))
        changed["image"][6][6] = [11, 4]
        changed["carrying"] = [6, 0]
        a, b = robots.feature_vector(original), robots.feature_vector(changed)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.isfinite(a).all())
        self.assertFalse(np.array_equal(a, b))

    def test_cli_preserves_test_row_order_without_test_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train").mkdir()
            (root / "test_public").mkdir()
            train = [observation(0), observation(0), observation(1)]
            test = [observation(1), observation(0), observation(1)]
            (root / "train" / "observations.json").write_text(json.dumps(train))
            (root / "train" / "labels.json").write_text(json.dumps([2, 2, 4]))
            (root / "test_public" / "observations.json").write_text(json.dumps(test))
            output = root / "predictions.json"
            robots.main(["--dataset-root", str(root), "--output", str(output), "--epochs", "0"])
            self.assertEqual(json.loads(output.read_text()), [4, 2, 4])

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                         "torch/numpy not installed")
    def test_small_training_smoke(self):
        train = [observation(0, direction=i % 4) for i in range(24)]
        labels = [i % 4 for i in range(24)]
        test = [observation(0, direction=1), observation(0, direction=2)]
        args = robots.make_parser().parse_args(["--epochs", "2", "--hidden", "16", "--batch-size", "8"])
        defaults = robots.majority_actions(train, labels)
        predictions = robots.train_predict(train, labels, test, defaults, args, time.monotonic() + 30)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(type(action) is int and action in range(6) for action in predictions))


if __name__ == "__main__":
    unittest.main()
