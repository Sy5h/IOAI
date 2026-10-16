"""Train IOAI Field from the currently mounted config(s), with a wall-time budget.

Use with custom_model.py containing the supplied field_custom_model.py code.
Paste this training file into solution.ipynb and call main([]). The official
protected generator requires its supported Linux x86-64 CPython 3.13 runtime.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_model_class():
    # During grading the file must be named custom_model.py. This fallback only
    # supports direct development from this repository's solutions directory.
    # Direct CLI execution sets sys.path[0] to this script's directory, not the
    # official notebook working directory where custom_model.py is located.
    local_module = Path.cwd() / "custom_model.py"
    if local_module.is_file() and str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    try:
        from custom_model import CustomModel
    except ModuleNotFoundError as exc:
        if exc.name != "custom_model":
            raise
        from field_custom_model import CustomModel
    return CustomModel


def read_configs(config_dir):
    directory = Path(config_dir)
    multi, single = directory / "field_configs.json", directory / "field_config.json"
    if multi.exists():
        raw = json.loads(multi.read_text(encoding="utf-8"))
    elif single.exists():
        raw = [json.loads(single.read_text(encoding="utf-8"))]
    else:
        raise FileNotFoundError(f"No field_config(s).json in {directory}")
    if not isinstance(raw, list) or not raw or not all(isinstance(x, dict) for x in raw):
        raise ValueError("Expected a nonempty list of field configuration objects.")
    return raw


def collect_regions(make_batch, cfg, n_each, seed):
    names = ("I", "O", "A", "I_entropy", "bg")
    regions = []
    for index, name in enumerate(names):
        xy, values = make_batch(n_each, cfg, include_regions=(name,),
                                stratified=True, seed=seed + index)
        xy, values = np.asarray(xy), np.asarray(values).reshape(-1)
        if xy.shape != (len(values), 2) or len(values) < 16:
            raise ValueError(f"Insufficient or malformed generated samples for {name}")
        regions.append((torch.as_tensor(xy, dtype=torch.float32),
                        torch.as_tensor(values, dtype=torch.float64).reshape(-1, 1)))
    return regions


def fit_affine_i(model, xy, values):
    """Fit the statement's linear gradient after scaling its huge target values."""
    scale = model.i_scale.detach().cpu().double()
    normalized = (values.cpu().double() - scale[0]) / scale[1]
    design = torch.cat([xy.cpu().double(), torch.ones(len(xy), 1, dtype=torch.float64)], 1)
    fit = torch.linalg.lstsq(design, normalized).solution
    with torch.no_grad():
        model.i_part.weight.copy_(fit[:2].T.to(model.i_part.weight))
        model.i_part.bias.copy_(fit[2].reshape_as(model.i_part.bias).to(model.i_part.bias))
    residual = (design @ fit - normalized).abs().mean().item()
    return residual


def calibrate_i_threshold(model, validation_regions, safety=0.5):
    """Require a margin above every observed non-I example; not a guarantee."""
    device = next(model.parameters()).device
    xy = torch.cat([pair[0] for pair in validation_regions[1:]], 0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model.router(xy)
        other = logits.clone()
        other[:, model.I] = -torch.inf
        largest_false_margin = (logits[:, model.I] - other.max(1).values).max().item()
    model.i_threshold = max(0.0, largest_false_margin + safety)
    return model.i_threshold


def train_router(model, regions, deadline, max_steps=2500, batch_size=2048):
    device = next(model.parameters()).device
    x = torch.cat([pair[0] for pair in regions], 0).to(device)
    labels = torch.cat([torch.full((len(pair[0]),), i, dtype=torch.long)
                        for i, pair in enumerate(regions)]).to(device)
    optimizer = torch.optim.Adam(model.router.parameters(), lr=0.002)
    model.router.train()
    steps = 0
    for step in range(max_steps):
        if time.monotonic() >= deadline:
            break
        idx = torch.randint(len(x), (min(batch_size, len(x)),), device=device)
        loss = F.cross_entropy(model.router(x[idx]), labels[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps = step + 1
    return steps


def train_o(model, samples, deadline, max_steps=2500, batch_size=2048):
    device = next(model.parameters()).device
    x, target = [t.to(device=device, dtype=torch.float32) for t in samples]
    optimizer = torch.optim.Adam(model.o_part.parameters(), lr=0.002)
    model.o_part.train()
    steps = 0
    for step in range(max_steps):
        if time.monotonic() >= deadline:
            break
        idx = torch.randint(len(x), (min(batch_size, len(x)),), device=device)
        prediction = torch.tanh(model.o_part(x[idx]))
        loss = F.smooth_l1_loss(prediction, target[idx], beta=0.05)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps = step + 1
    return steps


def train_one_config(cfg, make_batch, model_class, device, deadline,
                     samples_per_region=4096, validation_per_region=1024, seed=2026):
    torch.manual_seed(seed)
    i_min, i_max = float(cfg.i_grad_min), float(cfg.i_grad_max)
    if not i_max > i_min:
        raise ValueError("Expected i_grad_max > i_grad_min")
    model = model_class(i_min=i_min, i_max=i_max).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count >= 20260 or list(model.buffers()):
        raise ValueError("Model must have <20260 parameters and no buffers.")
    regions = collect_regions(make_batch, cfg, samples_per_region, seed)
    validation = collect_regions(make_batch, cfg, validation_per_region, seed + 100_000)
    affine_mae = fit_affine_i(model, *regions[0])
    remaining = max(0.0, deadline - time.monotonic())
    router_deadline = time.monotonic() + remaining * 0.62
    router_steps = train_router(model, regions, router_deadline)
    threshold = calibrate_i_threshold(model, validation)
    o_steps = train_o(model, regions[1], deadline)
    print(f"params={parameter_count}; I normalized MAE={affine_mae:.3g}; "
          f"router_steps={router_steps}; O_steps={o_steps}; I threshold={threshold:.3g}",
          flush=True)
    if affine_mae > 0.01:
        print("Warning: first-I affine fit residual is high; inspect this config.", flush=True)
    if router_steps == 0 or o_steps == 0:
        print("Warning: time budget expired before a component was trained.", flush=True)
    return model.eval().cpu()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="data/train_config")
    parser.add_argument("--output", default="model.pt")
    parser.add_argument("--seconds", type=float, default=265)
    parser.add_argument("--samples-per-region", type=int, default=4096)
    parser.add_argument("--validation-per-region", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    start = time.monotonic()
    try:
        from core import FieldConfig, make_batch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The official protected IOAI Field generator is required. Use the "
            "provided Linux x86-64 CPython 3.13 runtime and put grading-original "
            "on PYTHONPATH. This script does not replace or reconstruct the generator."
        ) from exc
    model_class = load_model_class()
    raw_configs = read_configs(args.config_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deadline = start + args.seconds
    models = []
    for i, raw in enumerate(raw_configs):
        # The grader remounts a hidden configuration and retrains this same code.
        remaining = max(0.0, deadline - time.monotonic() - 5.0)
        local_deadline = time.monotonic() + remaining / (len(raw_configs) - i)
        cfg = FieldConfig(**raw)
        model = train_one_config(
            cfg, make_batch, model_class, device, local_deadline,
            samples_per_region=args.samples_per_region,
            validation_per_region=args.validation_per_region,
            seed=args.seed + 1000 * i,
        )
        models.append(model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(models[0] if len(models) == 1 else models, output)
    print(f"Saved {output}; configs={len(models)}; elapsed={time.monotonic() - start:.1f}s")
    return models


if __name__ == "__main__" and "get_ipython" not in globals():
    main()
