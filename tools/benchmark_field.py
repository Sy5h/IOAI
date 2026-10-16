"""Reproduce IOAI Field on its supported runtime; tune only on public holdouts.

Public-data revisions and the original repository implementation are pinned.
Only already published JSON configuration files are downloaded from the dataset.
The protected Linux x86-64 CPython 3.13 generator is used without modification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

OFFICIAL_REV = "e1963801aa4cf33a5a03520b39fa6620a7894c54"
DATA_REV = "35136ba0abb05ea8dcdb6ec3db200617a9da6104"
BASELINE_REV = "b3e60ae8e14b55d97c44214dc1ec7d8e49295a89"
DATA_REPO = "IOAI-official/ioai-2026-ioai-field"
REPO = Path(__file__).resolve().parents[1]
SPLITS = {"public": "public/train_config", "released_leaderboard_a": "private/leaderboard_a_config",
          "released_leaderboard_b": "private/leaderboard_b_config"}
VALIDATION_SEEDS = (1729, 8071)
FIXED_STEPS = 2500
VALIDATION_POINTS_PER_REGION = 8192
CALIBRATION_OPTIONS = ((1024, 0.5), (16384, 1.5), (65536, 2.5))


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "IOAI-public-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def fetch(url, target, provenance):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        target.write_bytes(get(url))
    content = target.read_bytes()
    provenance.append({"url": url, "path": str(target), "bytes": len(content),
                       "sha256": hashlib.sha256(content).hexdigest()})


def prepare(work):
    provenance = []
    tree = json.loads(get(f"https://api.github.com/repos/IOAI-official/IOAI-2026/git/trees/{OFFICIAL_REV}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("Official file tree truncated")
    prefix = "Individual-Contest/6_IOAI_Field/code/"
    for entry in tree["tree"]:
        path = entry["path"]
        if entry["type"] != "blob" or not path.startswith(prefix):
            continue
        relative = path[len(prefix):]
        if not relative.startswith(("grading-original/", "baseline-original/")):
            continue
        if "__pycache__" in relative or "/tests/" in relative:
            continue
        url = f"https://raw.githubusercontent.com/IOAI-official/IOAI-2026/{OFFICIAL_REV}/{urllib.parse.quote(path)}"
        fetch(url, work / "official" / relative, provenance)
    entries = json.loads(get(f"https://huggingface.co/api/datasets/{DATA_REPO}/tree/{DATA_REV}?recursive=true&expand=false"))
    for entry in entries:
        path = entry["path"]
        if entry["type"] != "file" or not any(path.startswith(p + "/") for p in SPLITS.values()):
            continue
        if not path.endswith(("field_config.json", "field_configs.json", "eval_config.json")):
            continue
        url = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REV}/{urllib.parse.quote(path)}"
        fetch(url, work / "dataset" / path, provenance)
    for filename in ("day2_ioai_field.py", "field_custom_model.py"):
        url = f"https://raw.githubusercontent.com/Sy5h/IOAI/{BASELINE_REV}/solutions/{filename}"
        fetch(url, work / "repository-baseline" / filename, provenance)
    config_dirs = {name: work / "dataset" / path for name, path in SPLITS.items()}
    for directory in config_dirs.values():
        if not any((directory / name).is_file() for name in ("field_config.json", "field_configs.json")):
            raise FileNotFoundError(f"Pinned released field configuration missing: {directory.name}")
        if not (directory / "eval_config.json").is_file():
            raise FileNotFoundError(f"Pinned released evaluation configuration missing: {directory.name}")
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return config_dirs


def execute(command, cwd, env, logfile, timeout):
    started = time.monotonic()
    with logfile.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        try:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout)
            exit_code, timed_out = result.returncode, False
        except subprocess.TimeoutExpired:
            exit_code, timed_out = None, True
    print(f"--- {logfile.parent.name}/{logfile.name}: final 15 lines ---", flush=True)
    print("\n".join(logfile.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]), flush=True)
    return {"command": command, "wall_seconds": time.monotonic() - started,
            "exit_code": exit_code, "timed_out": timed_out}


def read_config_values(config_dir):
    multiple = config_dir / "field_configs.json"
    return json.loads(multiple.read_text()) if multiple.is_file() else [json.loads((config_dir / "field_config.json").read_text())]


def score(model_path, config_dir, output, eval_seed=None, n_per_region=None):
    import torch
    from core import FieldConfig, EvalConfig, evaluate_model
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    sys.path.insert(0, str(model_path.parent))
    configs = [FieldConfig(**raw) for raw in read_config_values(config_dir)]
    eval_values = json.loads((config_dir / "eval_config.json").read_text())
    if eval_seed is not None:
        eval_values["seed"] = eval_seed
    if n_per_region is not None:
        eval_values["n_per_region"] = n_per_region
    # Explicit dropout seed makes the stochastic baseline reproducible too.
    torch_seed = 2026 + int(eval_values["seed"])
    torch.manual_seed(torch_seed)
    started = time.monotonic()
    models = torch.load(model_path, map_location="cpu", weights_only=False)
    models = models if isinstance(models, list) else [models]
    if len(models) != len(configs):
        raise ValueError("Saved model count differs from mounted configuration count")
    scores = [evaluate_model(model, cfg, EvalConfig(**eval_values)) for model, cfg in zip(models, configs)]
    result = scores[0] if len(scores) == 1 else {
        "total_score": sum(item["total_score"] for item in scores) / len(scores), "per_config": scores}
    report = {"score": result, "evaluation_config": eval_values, "torch_seed": torch_seed,
              "configuration_count": len(configs), "points_per_region_per_config": eval_values["n_per_region"],
              "scoring_seconds": time.monotonic() - started,
              "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                           "cpu": platform.processor(), "cpu_count": os.cpu_count(),
                           "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                           "device_used": "CPU", "torch_threads": torch.get_num_threads()}}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(result.get("__explain__", result), flush=True)


def train_model(work, run_dir, kind, config_dir, steps=None, calibration=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config_dir, run_dir / "data/train_config", dirs_exist_ok=True)
    env = os.environ.copy()
    env.update({"PYTHONPATH": os.pathsep.join((str(run_dir), str(work / "official/grading-original"))),
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "MPLBACKEND": "Agg",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if kind == "official_baseline":
        baseline = work / "official/baseline-original"
        model_source = baseline / "custom_model.py"
        notebook = json.loads((baseline / "solution.ipynb").read_text(encoding="utf-8"))
        source = "\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        training_source = run_dir / "train.py"
        training_source.write_text(source, encoding="utf-8")
        command = [sys.executable, "train.py"]
    else:
        source_dir = work / "repository-baseline" if kind == "repository_original" else REPO / "solutions"
        model_source = source_dir / "field_custom_model.py"
        training_source = source_dir / "day2_ioai_field.py"
        command = [sys.executable, str(training_source), "--seconds", "265"]
        if steps is not None:
            command += ["--router-steps", str(steps), "--o-steps", str(steps)]
        if calibration is not None:
            command += ["--validation-per-region", str(calibration[0]), "--i-safety", str(calibration[1])]
    shutil.copy2(model_source, run_dir / "custom_model.py")
    result = {"training_source_sha256": hashlib.sha256(training_source.read_bytes()).hexdigest(),
              "model_source_sha256": hashlib.sha256(model_source.read_bytes()).hexdigest(),
              "router_steps_requested": steps, "o_steps_requested": steps,
              "calibration": None if calibration is None else {"validation_per_region": calibration[0], "i_safety": calibration[1]},
              "configuration_count": len(read_config_values(config_dir)),
              "training": execute(command, run_dir, env, run_dir / "training.log", timeout=350)}
    return result, env


def score_run(run_dir, config_dir, env, label="score", eval_seed=None, n_per_region=None):
    output = run_dir / (label + ".json")
    command = [sys.executable, str(Path(__file__).resolve()), "--score-model", str(run_dir / "model.pt"),
               "--config-dir", str(config_dir), "--output", str(output)]
    if eval_seed is not None:
        command += ["--eval-seed", str(eval_seed)]
    if n_per_region is not None:
        command += ["--n-per-region", str(n_per_region)]
    timing = execute(command, run_dir, env, run_dir / (label + ".log"), timeout=180)
    if timing["exit_code"] != 0 or not output.is_file():
        raise RuntimeError(f"Scoring failed: {run_dir.name}/{label}")
    return {"scoring": timing, "result": json.loads(output.read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="work/field-benchmark")
    parser.add_argument("--score-model")
    parser.add_argument("--config-dir")
    parser.add_argument("--output")
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument("--n-per-region", type=int)
    args = parser.parse_args()
    if args.score_model:
        score(Path(args.score_model), Path(args.config_dir), Path(args.output), args.eval_seed, args.n_per_region)
        return
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("This benchmark requires Linux x86-64 and CPython 3.13.")
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    overall_start = time.monotonic()
    config_dirs = prepare(work)
    summary = {"official_revision": OFFICIAL_REV, "dataset_revision": DATA_REV,
               "baseline_commit": BASELINE_REV, "device": "CPU",
               "note": "Development re-evaluation after observing earlier A/B errors; not independent confirmation or a hidden leaderboard submission.",
               "selection": {"split": "public/train_config", "validation_seeds": list(VALIDATION_SEEDS),
                             "fixed_steps": FIXED_STEPS, "validation_points_per_region": VALIDATION_POINTS_PER_REGION,
                             "calibration_options": list(CALIBRATION_OPTIONS), "candidates": {}}, "runs": {}}
    def save():
        (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # This follow-up was motivated by earlier A/B errors. Candidate choice itself
    # uses public generated holdouts only; final scores are development re-tests.
    for calibration in CALIBRATION_OPTIONS:
        key = f"n{calibration[0]}_s{calibration[1]}"
        run_dir = work / f"selection_{key}"
        report, env = train_model(work, run_dir, "reliability_candidate", config_dirs["public"], FIXED_STEPS, calibration)
        if report["training"]["exit_code"] != 0:
            summary["selection"]["candidates"][key] = report
            save()
            raise RuntimeError("Candidate training failed; inspect training log")
        report["validation"] = {str(seed): score_run(run_dir, config_dirs["public"], env,
                                  label=f"validation_{seed}", eval_seed=seed,
                                  n_per_region=VALIDATION_POINTS_PER_REGION) for seed in VALIDATION_SEEDS}
        report["mean_validation_score"] = sum(x["result"]["score"]["total_score"]
                                                   for x in report["validation"].values()) / len(VALIDATION_SEEDS)
        summary["selection"]["candidates"][key] = report
        print("SELECTION", key, json.dumps(report), flush=True)
        save()
    selected = max(CALIBRATION_OPTIONS, key=lambda c: (
        summary["selection"]["candidates"][f"n{c[0]}_s{c[1]}"]["mean_validation_score"], -c[0]))
    summary["selection"]["selected_calibration"] = {"validation_per_region": selected[0], "i_safety": selected[1]}
    print("SELECTED_CALIBRATION", json.dumps(summary["selection"]["selected_calibration"]), flush=True)
    save()
    for split, config_dir in config_dirs.items():
        summary["runs"][split] = {}
        for name in ("official_baseline", "repository_original", "reliability_candidate"):
            run_dir = work / f"{split}_{name}"
            report, env = train_model(work, run_dir, name, config_dir,
                                      steps=FIXED_STEPS if name == "reliability_candidate" else None,
                                      calibration=selected if name == "reliability_candidate" else None)
            summary["runs"][split][name] = report
            if report["training"]["exit_code"] == 0 and (run_dir / "model.pt").is_file():
                report.update(score_run(run_dir, config_dir, env))
            print("FINAL_RESULT", split, name, json.dumps(report), flush=True)
            save()
    summary["total_setup_training_scoring_seconds"] = time.monotonic() - overall_start
    save()
    if any("result" not in r for split in summary["runs"].values() for r in split.values()):
        raise SystemExit("At least one model failed; inspect uploaded training/scoring logs.")


if __name__ == "__main__":
    main()
