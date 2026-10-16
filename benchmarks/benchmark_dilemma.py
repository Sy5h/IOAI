"""Measure a local solution using the unmodified official public Dilemma scorer.

Example:
python benchmarks/benchmark_dilemma.py --assets /path/to/work/bench-vision/dilemma \
  --source solutions/day2_double_agent_dilemma.py --split train --limit 20 \
  --seconds 132 --output /path/to/work/bench-vision/train-current
"""
import time
STARTED = time.monotonic()
import argparse
import contextlib
import hashlib
import importlib.util
import io
import inspect
import json
import os
from pathlib import Path
import platform
import sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--deps")
    parser.add_argument("--split", choices=["train", "test_public", "test_leaderboard_a", "test_leaderboard_b"], default="test_public")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seconds", type=float, default=660)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--first-attempt-fraction", type=float)
    parser.add_argument("--check-clean", action="store_true")
    args = parser.parse_args()
    if args.deps:
        sys.path.insert(0, str(Path(args.deps).resolve()))
    import torch
    import torchvision
    import timm
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    root, source, output = Path(args.assets).resolve(), Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MODELS_DIR"] = str(root / "models")
    module = load_module("tested_dilemma_solution", source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_directory = "private" if args.split.startswith("test_leaderboard_") else "public"
    targets = module.load_targets(root / data_directory, args.split)
    if args.limit is not None:
        targets = targets[:args.limit]
    # Clear predictions are a model/preprocessing compatibility check, outside
    # the measured attack run. Both official models should be correct on all data.
    correct_r = correct_v = None
    if args.check_clean:
        r, v = module.load_models(root / "models", device)
        preprocess = module.make_preprocess(device)
        correct_r = correct_v = 0
        with torch.no_grad():
            for item in targets:
                image = to_tensor(Image.open(item["image_path"]).convert("RGB")).to(device)
                batch = preprocess(image).unsqueeze(0)
                correct_r += int(r(batch).argmax(-1).item() == item["label"])
                correct_v += int(v(batch).argmax(-1).item() == item["label"])
        print(f"Clean model compatibility: ResNet={correct_r}/{len(targets)}, ViT={correct_v}/{len(targets)}", flush=True)
        if correct_r != len(targets) or correct_v != len(targets):
            raise RuntimeError("Downloaded weights/preprocessing do not reproduce clean 100% accuracy")
        del r, v
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # Without --check-clean, count imports and first CUDA/model initialization.
    attack_start = time.monotonic() if args.check_clean else STARTED
    r, v = module.load_models(root / "models", device)
    extra = {} if args.first_attempt_fraction is None else {"first_attempt_fraction": args.first_attempt_fraction}
    archive = module.run_attacks(targets, r, v, output / "submission.zip",
                                 seconds=args.seconds, started_at=attack_start,
                                 max_steps=args.max_steps, learning_rate=args.learning_rate,
                                 device=device, **extra)
    if device.type == "cuda":
        torch.cuda.synchronize()
    run_seconds = time.monotonic() - attack_start
    grader_path = root / "grading-original/evaluate.py"
    grader = load_module("official_dilemma_evaluate", grader_path)
    # Cache the identical fixed weights; no scorer logic is modified.
    grader._resnet, grader._vit = r, v
    score_start = time.monotonic()
    predictions = grader.build_preds(archive, targets)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        score = grader.compute_score(predictions, targets, verbose=True)
    scoring_seconds = time.monotonic() - score_start
    (output / "scorer.log").write_text(stdout.getvalue(), encoding="utf-8")
    result = {
        "task": "Double Agent Dilemma", "baseline_commit": "b3e60ae8e14b55d97c44214dc1ec7d8e49295a89",
        "dataset_revision": "6f42de8f02c801618d0b90189bf4c759a9cd6381",
        "split": args.split, "count": len(targets), "metric": "official pure_success_fraction * sigmoid_L2_penalty",
        "score_0_to_1": score, "score_0_to_100": 100 * score, "run_seconds": run_seconds,
        "scoring_seconds": scoring_seconds, "whole_process_seconds": time.monotonic() - STARTED,
        "requested_seconds": args.seconds, "max_steps": args.max_steps, "learning_rate": args.learning_rate,
        "first_attempt_fraction": args.first_attempt_fraction if args.first_attempt_fraction is not None else
            (inspect.signature(module.run_attacks).parameters["first_attempt_fraction"].default
             if "first_attempt_fraction" in inspect.signature(module.run_attacks).parameters else None),
        "timing_includes_imports_and_first_cuda_init": not args.check_clean,
        "fixed_model_parameters": {"resnet18": sum(p.numel() for p in r.parameters()),
                                   "vit_tiny_patch16_224": sum(p.numel() for p in v.parameters())},
        "clean_accuracy": None if correct_r is None else {"resnet": correct_r / len(targets), "vit": correct_v / len(targets)},
        "source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "grader_sha256": hashlib.sha256(grader_path.read_bytes()).hexdigest(),
        "command": [sys.executable] + sys.argv, "hardware": {
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
            "cpu": platform.processor(), "torch_threads": torch.get_num_threads(),
            "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        },
    }
    manifest_path = root.parent / "download-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result["official_source_tree_sha"] = manifest.get("github_tree_sha")
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
