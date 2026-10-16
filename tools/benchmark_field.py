"""Reproduce the public IOAI Field score on the official supported CPU runtime.

Downloads public configuration and unmodified protected runtime from pinned
revisions. No private split, secrets, paid resources, or protection bypasses.
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
DATA_REPO = "IOAI-official/ioai-2026-ioai-field"
REPO = Path(__file__).resolve().parents[1]


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
        if entry["type"] != "file" or not path.startswith("public/train_config/"):
            continue
        url = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REV}/{urllib.parse.quote(path)}"
        fetch(url, work / "dataset" / path, provenance)
    config_dir = work / "dataset/public/train_config"
    if not (config_dir / "field_config.json").is_file():
        raise FileNotFoundError("Pinned PUBLIC field configuration missing")
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return config_dir


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
    return {"command": command, "wall_seconds": time.monotonic() - started,
            "exit_code": exit_code, "timed_out": timed_out}


def score(model_path, config_dir, output):
    import torch
    from core import FieldConfig, EvalConfig, evaluate_model
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    # torch.load needs the same canonical custom_model module used in training.
    sys.path.insert(0, str(model_path.parent))
    cfg = FieldConfig(**json.loads((config_dir / "field_config.json").read_text()))
    eval_path = config_dir / "eval_config.json"
    eval_values = json.loads(eval_path.read_text()) if eval_path.is_file() else {
        "n_per_region": 512, "seed": 2026, "range_mode": "p99", "entropy_runs": 10,
        "entropy_mode": "std", "enforce_unit_interval": True,
        "entropy_min": -2026, "entropy_max": 2026, "max_params": 20260, "param_penalty": 0.5,
    }
    started = time.monotonic()
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    result = evaluate_model(model, cfg, EvalConfig(**eval_values))
    report = {"score": result, "evaluation_config": eval_values,
              "scoring_seconds": time.monotonic() - started,
              "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                           "cpu": platform.processor(), "cpu_count": os.cpu_count(),
                           "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                           "device_used": "CPU", "torch_threads": torch.get_num_threads()}}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(result.get("__explain__", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="work/field-benchmark")
    parser.add_argument("--score-model")
    parser.add_argument("--config-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.score_model:
        score(Path(args.score_model), Path(args.config_dir), Path(args.output))
        return
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("This benchmark requires Linux x86-64 and CPython 3.13.")
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    overall_start = time.monotonic()
    config_dir = prepare(work)
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(work / "official/grading-original"),
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "MPLBACKEND": "Agg",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    summary = {"official_revision": OFFICIAL_REV, "dataset_revision": DATA_REV,
               "split": "public/train_config", "device": "CPU",
               "note": "Public-config CPU reproduction; not contest GPU or hidden leaderboard results.",
               "runs": {}}
    for name in ("official_baseline", "repository_current"):
        run_dir = work / name
        run_dir.mkdir(exist_ok=True)
        shutil.copytree(config_dir, run_dir / "data/train_config", dirs_exist_ok=True)
        if name == "official_baseline":
            baseline = work / "official/baseline-original"
            shutil.copy2(baseline / "custom_model.py", run_dir / "custom_model.py")
            notebook = json.loads((baseline / "solution.ipynb").read_text(encoding="utf-8"))
            source = "\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
            (run_dir / "train.py").write_text(source, encoding="utf-8")
            command = [sys.executable, "train.py"]
        else:
            shutil.copy2(REPO / "solutions/field_custom_model.py", run_dir / "custom_model.py")
            command = [sys.executable, str(REPO / "solutions/day2_ioai_field.py"), "--seconds", "265"]
        training = execute(command, run_dir, env, run_dir / "training.log", timeout=350)
        summary["runs"][name] = {"training": training}
        model_path = run_dir / "model.pt"
        if training["exit_code"] == 0 and model_path.is_file():
            result_path = run_dir / "score.json"
            scoring = execute([sys.executable, str(Path(__file__).resolve()), "--score-model", str(model_path),
                               "--config-dir", str(config_dir), "--output", str(result_path)],
                              run_dir, env, run_dir / "scoring.log", timeout=180)
            summary["runs"][name]["scoring"] = scoring
            if result_path.is_file():
                summary["runs"][name]["result"] = json.loads(result_path.read_text())
        print(name, json.dumps(summary["runs"][name]), flush=True)
        (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["total_setup_training_scoring_seconds"] = time.monotonic() - overall_start
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if any("result" not in x for x in summary["runs"].values()):
        raise SystemExit("At least one model failed; inspect uploaded training/scoring logs.")


if __name__ == "__main__":
    main()
