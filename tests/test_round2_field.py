"""Synthetic geometry contract tests, not an emulation of the protected generator."""
import ast
import io
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

SOLUTIONS = Path(__file__).resolve().parents[1] / "solutions"
sys.path.insert(0, str(SOLUTIONS))
import day2_ioai_field as train
from field_custom_model import CustomModel


class AllI(nn.Module):
    def forward(self, xy):
        values = torch.zeros(len(xy), 5, device=xy.device)
        values[:, 0] = 100.
        values[:, 4] = 1.
        return values


def sample_toy(seed, count):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-.6, .6, (count, 2))
    inside = ((np.abs(raw[:, 0]) < .08) & (np.abs(raw[:, 1]) < .4)) | (
        (np.abs(raw[:, 0]) < .25) & (np.abs(raw[:, 1]) > .3) & (np.abs(raw[:, 1]) < .4))
    angle = .9
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = torch.tensor(raw @ rotation.T + .5, dtype=torch.float32)
    return xy, inside, rotation


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sample_hulls_preserve_rotated_concavity_and_ignore_bad_mlp(self):
        xy, positive, rotation = sample_toy(11, 60000)
        held, held_positive, _ = sample_toy(29, 60000)
        for axis in ("affine", "pca"):
            with self.subTest(axis=axis):
                model = CustomModel()
                with torch.no_grad():
                    model.i_part.weight.copy_(torch.tensor(rotation[:, 1]).float()[None])
                report = train.fit_i_geometry(model, xy[positive], xy[~positive], bands=32, axis=axis)
                self.assertGreater(report["accepted_pieces"], 20)
                self.assertLess(sum(p.numel() for p in model.parameters()), 20260)
                self.assertEqual(list(model.buffers()), [])
                model.router = AllI()
                model.eval()
                selected = model.regions(held) == model.I
                self.assertEqual(selected[~held_positive].sum().item(), 0)
                self.assertGreater(selected[held_positive].float().mean().item(), .85)
                model.train()
                torch.testing.assert_close(model.regions(held) == model.I, selected)

    def test_negative_point_vetoes_a_concave_bridge(self):
        model = CustomModel()
        with torch.no_grad():
            model.i_part.weight.copy_(torch.tensor([[0., 1.]]))
        vertices = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        report = train.fit_i_geometry(model, vertices, torch.tensor([[.5, .5]]), bands=1)
        self.assertEqual(report["rejected_pieces"], 1)
        self.assertFalse(model.i_geometry(vertices).any().item())

    def test_geometry_serializes_as_parameters_without_non_torch_imports(self):
        model = CustomModel()
        model.set_i_geometry([np.array([[1., 0., -.8], [-1., 0., .2],
                                       [0., 1., -.8], [0., -1., .2]])])
        stream = io.BytesIO()
        torch.save(model, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        self.assertEqual(list(restored.buffers()), [])
        torch.testing.assert_close(restored.i_geometry(torch.tensor([[.5, .5], [1., 1.]])),
                                   torch.tensor([True, False]))
        for node in ast.walk(ast.parse((SOLUTIONS / "field_custom_model.py").read_text())):
            if isinstance(node, ast.Import):
                self.assertTrue(all(item.name.split(".")[0] == "torch" for item in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.module.split(".")[0], "torch")

    def test_selected_defaults_match_the_benchmarked_explicit_cli(self):
        signature = inspect.signature(train.train_one_config)
        self.assertEqual(signature.parameters["i_geometry_bands"].default, 32)
        self.assertEqual(signature.parameters["i_geometry_axis"].default, "pca")
        core_stub = types.SimpleNamespace(FieldConfig=lambda **kwargs: kwargs, make_batch=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "field_config.json").write_text(json.dumps({"i_grad_min": 1e10, "i_grad_max": 1e20}))
            base = ["--config-dir", directory, "--output", str(root / "model.pt")]
            with patch.dict(sys.modules, {"core": core_stub}), \
                 patch.object(train.torch.cuda, "is_available", return_value=False), \
                 patch.object(train, "train_one_config", return_value=nn.Linear(2, 1)) as fit, \
                 patch.object(train.torch, "save"), patch("sys.stdout", new_callable=io.StringIO):
                train.main(base)
                default_kwargs = fit.call_args.kwargs.copy()
                train.main(base + ["--i-geometry-bands", "32", "--i-geometry-axis", "pca"])
                self.assertEqual(default_kwargs, fit.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
