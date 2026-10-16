"""Synthetic contract tests; these do not claim official dataset scores."""

import ast
import io
import json
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SOLUTION_DIR = Path(__file__).resolve().parents[1] / "solutions"
sys.path.insert(0, str(SOLUTION_DIR))
import day2_double_agent_dilemma as attack
import day2_ioai_field as field_train
from field_custom_model import CustomModel, DropoutField


class PixelClassifier(nn.Module):
    def __init__(self, channel, threshold):
        super().__init__()
        self.channel = channel
        self.threshold = threshold

    def forward(self, x):
        logit = 4 * (x[:, self.channel, 0, 0] - self.threshold)
        return torch.stack([logit, torch.zeros_like(logit)], 1)


class RuleRouter(nn.Module):
    def forward(self, xy):
        index = (5 * xy[:, 0]).long().clamp(0, 4)
        return torch.zeros(len(xy), 5, device=xy.device).scatter_(1, index[:, None], 10.)


class AttackTests(unittest.TestCase):
    def test_cleanup_preserves_actual_pixels_and_reduces_norm(self):
        torch.manual_seed(42)
        image = torch.rand(3, 7, 5)
        delta = 3 * torch.randn_like(image)
        clean = attack.effective_delta(image, delta)
        torch.testing.assert_close((image + delta).clamp(0, 1),
                                   (image + clean).clamp(0, 1))
        self.assertLessEqual(clean.norm().item(), delta.norm().item())

    def test_attack_and_shrink_keep_verified_differential_success(self):
        image = torch.full((3, 2, 2), 0.5)
        keep, fool = PixelClassifier(0, 0.2), PixelClassifier(1, 0.49)
        preprocess = lambda value: value
        delta = attack.attack_one(image, 0, keep, fool, preprocess, max_steps=30,
                                  learning_rate=0.01, initial_epsilon=0.1,
                                  norm_weight=0.0, shrink_steps=6)
        self.assertTrue(attack.is_success(image, delta, 0, keep, fool, preprocess))
        self.assertLess(delta.norm().item(), 0.02)
        self.assertEqual(torch.count_nonzero(delta[0]).item(), 0)

    def test_impossible_attack_and_expired_budget_return_zero(self):
        image = torch.full((3, 2, 2), 0.5)
        model = PixelClassifier(0, 0.49)
        delta = attack.attack_one(image, 0, model, model, lambda x: x, max_steps=5)
        self.assertEqual(torch.count_nonzero(delta).item(), 0)
        expired = attack.attack_one(image, 0, model, model, lambda x: x,
                                    deadline=time.monotonic() - 1)
        self.assertEqual(torch.count_nonzero(expired).item(), 0)

    def test_flat_zip_roundtrip(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            attack.write_pair(archive, 12, torch.zeros(3, 7, 5), torch.ones(3, 7, 5))
        with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
            self.assertEqual(set(archive.namelist()), {"12_a.pt", "12_b.pt"})
            tensor = torch.load(io.BytesIO(archive.read("12_b.pt")), weights_only=True)
            self.assertEqual(tensor.shape, (3, 7, 5))
            self.assertTrue(torch.all(tensor == 1))


class FieldTests(unittest.TestCase):
    def test_dropout_head_bounds_levels_and_exact_metric(self):
        torch.manual_seed(2026)
        head = DropoutField().train()
        xy = torch.zeros(5000, 2)
        with torch.no_grad():
            values = torch.stack([head(xy).squeeze(1) for _ in range(10)]).numpy()
        self.assertGreaterEqual(float(values.min()), -2026.)
        self.assertLessEqual(float(values.max()), 2026.)
        self.assertEqual(len(np.unique(values)), 16)
        scores = np.minimum(values.std(axis=0, ddof=0) / np.sqrt(1.0 / 12.0), 1.)
        self.assertEqual(float(scores.mean()), 1.)
        head.eval()
        torch.testing.assert_close(head(xy), torch.zeros(5000, 1))

    def test_parameter_budget_buffers_and_torch_only_import_contract(self):
        model = CustomModel()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 17424)
        self.assertLess(sum(p.numel() for p in model.parameters()), 20260)
        self.assertEqual(list(model.buffers()), [])
        tree = ast.parse((SOLUTION_DIR / "field_custom_model.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(name.name.split(".")[0] == "torch" for name in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.module.split(".")[0], "torch")

    def test_hard_routing_constants_and_dropout_only_in_entropy_region(self):
        model = CustomModel(i_min=10., i_max=20.)
        model.router = RuleRouter()
        with torch.no_grad():
            model.i_part.weight.zero_()
            model.i_part.bias.fill_(0.25)
            for p in model.o_part.parameters():
                p.zero_()
        xy = torch.tensor([[0.1, 0.2], [0.3, 0.2], [0.5, 0.2],
                           [0.7, 0.2], [0.9, 0.2]])
        model.eval()
        expected = torch.tensor([[12.5], [0.], [-1.], [0.], [0.]])
        torch.testing.assert_close(model(xy), expected)
        model.train()
        for _ in range(10):
            result = model(xy)
            torch.testing.assert_close(result[[0, 1, 2, 4]], expected[[0, 1, 2, 4]])
            self.assertLessEqual(abs(result[3].item()), 1920.)

    def test_affine_fit_recovers_normalized_linear_gradient(self):
        torch.manual_seed(9)
        model = CustomModel(i_min=1e10, i_max=2e10)
        xy = torch.rand(200, 2)
        normalized = 0.1 + 0.3 * xy[:, :1].double() + 0.5 * xy[:, 1:].double()
        target = 1e10 + 1e10 * normalized
        residual = field_train.fit_affine_i(model, xy, target)
        self.assertLess(residual, 1e-10)
        torch.testing.assert_close(model.i_part(xy), normalized.float(), atol=1e-6, rtol=1e-6)

    def test_multiple_config_loading_uses_mounted_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "field_config.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
            self.assertEqual(field_train.read_configs(root), [{"a": 1}])
            (root / "field_configs.json").write_text(json.dumps([{"a": 2}, {"a": 3}]),
                                                     encoding="utf-8")
            self.assertEqual(field_train.read_configs(root), [{"a": 2}, {"a": 3}])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
