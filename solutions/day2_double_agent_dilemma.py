"""IOAI 2026 task 4: per-image, white-box differential attacks.

Paste this entire file into an official solution.ipynb cell, then call main([]).
For CLI use, run with --help. No model downloads are attempted.
Public benchmark results and reproducible commands are recorded in benchmarks/.
"""

import argparse
import io
import json
import os
import time
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F


def effective_delta(image, delta):
    """Remove changes that clipping would discard, preserving the actual image."""
    return (image + delta).clamp(0.0, 1.0) - image


def margin(logits, label):
    """Positive exactly when the true-class logit beats every other class."""
    other = logits.clone()
    other[..., label] = -torch.inf
    return logits[..., label] - other.max(dim=-1).values


def make_preprocess(device):
    # Tensor resize preserves the input gradient and matches the official grader.
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    def preprocess(image):
        resized = TF.resize(image, size=256, interpolation=T.InterpolationMode.BILINEAR)
        return (TF.center_crop(resized, output_size=224) - mean) / std

    return preprocess


def is_success(image, delta, label, keep_model, fool_model, preprocess):
    with torch.no_grad():
        batch = preprocess((image + delta).clamp(0, 1)).unsqueeze(0)
        kept = keep_model(batch).argmax(dim=-1).item() == label
        fooled = fool_model(batch).argmax(dim=-1).item() != label
    return kept and fooled


def shrink_success(image, delta, label, keep_model, fool_model, preprocess,
                   deadline=float("inf"), steps=8):
    """Keep a verified successful candidate while testing smaller scales.

    Success along a ray need not be monotone; bisection is only a heuristic to
    propose candidates. The returned value is always one that was checked.
    """
    best = effective_delta(image, delta.detach())
    if not is_success(image, best, label, keep_model, fool_model, preprocess):
        return torch.zeros_like(image)
    base = best.clone()
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        if time.monotonic() >= deadline:
            break
        mid = (lo + hi) / 2
        candidate = effective_delta(image, base * mid)
        if is_success(image, candidate, label, keep_model, fool_model, preprocess):
            best, hi = candidate, mid
        else:
            lo = mid
    return best.detach()


def attack_one(image, label, keep_model, fool_model, preprocess,
               deadline=float("inf"), max_steps=60, learning_rate=0.0001,
               initial_epsilon=0.004, keep_weight=2.0, norm_weight=0.02,
               safety_margin=0.05, shrink_steps=8):
    """Optimize only a raw-resolution perturbation, returning zero on failure.

    Epsilon is a search control, not an official hard constraint. It grows every
    20 steps. All model weights should already be frozen and in eval mode.
    """
    delta = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)
    best = None
    for step in range(max_steps + 1):
        if time.monotonic() >= deadline:
            break
        with torch.enable_grad():
            batch = preprocess((image + delta).clamp(0, 1)).unsqueeze(0)
            keep_logits = keep_model(batch)
            fool_logits = fool_model(batch)
            success = (keep_logits.argmax(-1).item() == label
                       and fool_logits.argmax(-1).item() != label)
            if success:
                best = effective_delta(image, delta.detach())
                break
            if step == max_steps:
                break
            keep_margin = margin(keep_logits, label)
            fool_margin = margin(fool_logits, label)
            loss = (F.relu(fool_margin + safety_margin).mean()
                    + keep_weight * F.relu(safety_margin - keep_margin).mean()
                    + norm_weight * delta.square().sum())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            epsilon = initial_epsilon * (2 ** min(step // 20, 3))
            delta.clamp_(-epsilon, epsilon)
            delta.copy_(effective_delta(image, delta))
    # An unsuccessful nonzero perturbation only increases the global penalty.
    if best is None:
        return torch.zeros_like(image)
    if time.monotonic() >= deadline:
        return best.detach()
    return shrink_success(image, best, label, keep_model, fool_model, preprocess,
                          deadline=deadline, steps=shrink_steps)


def attack_with_fallback(image, label, keep_model, fool_model, preprocess,
                         deadline=float("inf"), max_steps=60,
                         learning_rate=0.0001, initial_epsilon=0.004,
                         first_attempt_fraction=0.65):
    """Try a small step first, then retry failures with the original larger step."""
    options = dict(deadline=deadline, max_steps=max_steps,
                   initial_epsilon=initial_epsilon)
    if not 0 < first_attempt_fraction <= 1:
        raise ValueError("first_attempt_fraction must be in (0, 1].")
    started = time.monotonic()
    # Reserve time for the larger-step retry instead of exhausting the local
    # deadline on the first attempt. A value of 1 reproduces the earlier policy.
    primary_deadline = min(deadline, started + first_attempt_fraction * max(0., deadline - started))
    if learning_rate >= 0.001:
        primary_deadline = deadline
    primary_options = dict(options, deadline=primary_deadline)
    delta = attack_one(image, label, keep_model, fool_model, preprocess,
                       learning_rate=learning_rate, **primary_options)
    if (learning_rate < 0.001 and not torch.count_nonzero(delta).item()
            and time.monotonic() < deadline):
        delta = attack_one(image, label, keep_model, fool_model, preprocess,
                           learning_rate=0.001, **options)
    return delta


def load_models(models_dir, device):
    import timm
    from torchvision import models

    root = Path(models_dir)
    resnet = models.resnet18(weights=None)
    resnet.load_state_dict(torch.load(root / "resnet18.pth", map_location="cpu",
                                     weights_only=True))
    vit = timm.create_model(
        "vit_tiny_patch16_224", pretrained=True,
        pretrained_cfg_overlay={
            "file": str(root / "vit_tiny_patch16_224.safetensors"),
            "custom_load": False,
        },
    )
    for model in (resnet, vit):
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return resnet, vit


def load_targets(dataset_dir, split):
    directory = Path(dataset_dir) / split
    with (directory / "labels.json").open(encoding="utf-8") as handle:
        labels = json.load(handle)
    targets = []
    for key, label in sorted(labels.items(), key=lambda pair: int(pair[0])):
        idx = int(key)
        path = directory / "images" / f"{idx:04d}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        targets.append({"idx": idx, "image_path": path, "label": int(label)})
    return targets


def write_pair(archive, idx, delta_a, delta_b):
    """Write the exact, flat, original-resolution output format."""
    for tag, tensor in (("a", delta_a), ("b", delta_b)):
        tensor = tensor.detach().cpu().float().contiguous()
        if tensor.ndim != 3 or tensor.shape[0] != 3 or not torch.isfinite(tensor).all():
            raise ValueError("Perturbations must be finite 3 x H x W tensors.")
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        archive.writestr(f"{idx}_{tag}.pt", buffer.getvalue())


def run_attacks(targets, resnet, vit, output="submission.zip", seconds=660,
                max_steps=60, learning_rate=0.0001, initial_epsilon=0.004,
                started_at=None, device=None, first_attempt_fraction=0.65):
    from PIL import Image
    import torchvision.transforms.functional as TF

    if device is None:
        device = next(resnet.parameters()).device
    preprocess = make_preprocess(device)
    started_at = time.monotonic() if started_at is None else started_at
    deadline = started_at + seconds
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    successes = 0
    # Streaming avoids keeping 200 full-resolution tensors in GPU/CPU memory.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for item in targets:
            image = TF.to_tensor(Image.open(item["image_path"]).convert("RGB")).to(device)
            pair = []
            for keep, fool in ((resnet, vit), (vit, resnet)):
                remaining_jobs = 2 * len(targets) - completed
                # Share remaining compute between all images. Reserve packing time.
                remaining = max(0.0, deadline - time.monotonic() - 10.0)
                allowance = remaining / max(1, remaining_jobs)
                local_deadline = time.monotonic() + allowance
                delta = attack_with_fallback(
                    image, item["label"], keep, fool, preprocess,
                    deadline=local_deadline, max_steps=max_steps,
                    learning_rate=learning_rate, initial_epsilon=initial_epsilon,
                    first_attempt_fraction=first_attempt_fraction,
                ) if allowance > 0.05 else torch.zeros_like(image)
                # Both attempts return a verified success or zero.
                successes += int(bool(torch.count_nonzero(delta).item()))
                pair.append(delta.cpu())
                completed += 1
            write_pair(archive, item["idx"], *pair)
            print(f"{completed}/{2 * len(targets)} attacks; verified successes={successes}",
                  flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "dataset"))
    parser.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "models"))
    parser.add_argument("--split", default=os.environ.get("TEST_SPLIT", "test_public"))
    parser.add_argument("--output", default="submission.zip")
    parser.add_argument("--seconds", type=float, default=660)
    parser.add_argument("--max-steps", type=int, default=60)
    # Selected on the first 20 public training images before test evaluation.
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--first-attempt-fraction", type=float, default=0.65)
    parser.add_argument("--epsilon", type=float, default=0.004)
    args = parser.parse_args(argv)
    start = time.monotonic()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = load_targets(args.data_dir, args.split)
    resnet, vit = load_models(args.models_dir, device)
    result = run_attacks(
        targets, resnet, vit, output=args.output, seconds=args.seconds,
        max_steps=args.max_steps, learning_rate=args.learning_rate,
        initial_epsilon=args.epsilon, started_at=start, device=device,
        first_attempt_fraction=args.first_attempt_fraction,
    )
    print(f"Saved {result}; elapsed={time.monotonic() - start:.1f}s")
    return result


if __name__ == "__main__" and "get_ipython" not in globals():
    main()
