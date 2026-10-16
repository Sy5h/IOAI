"""Round 2: sample-fitted I geometry, selected on public generated holdouts.

Run only on the official supported Linux x86-64 CPython 3.13 runtime. Uses the
unmodified official make_batch and evaluator; no protected implementation is
inspected or replaced. Released A/B results are development re-evaluations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import benchmark_field as common

BASELINE_REV = "991a49b48092286967b5bd7cada465e336ebc4af"
VALIDATION_SEEDS = (314159, 271828)
VALIDATION_POINTS = 8192
CANDIDATES = (("baseline_991a49b", 0, "affine"),
              ("hulls16_affine", 16, "affine"), ("hulls32_affine", 32, "affine"),
              ("hulls16_pca", 16, "pca"), ("hulls32_pca", 32, "pca"))


def train(work, run_dir, config_dir, candidate):
    key, bands, axis = candidate
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config_dir, run_dir / "data/train_config", dirs_exist_ok=True)
    baseline = key == "baseline_991a49b"
    source_dir = work / "repository-baseline" if baseline else common.REPO / "solutions"
    source, model = source_dir / "day2_ioai_field.py", source_dir / "field_custom_model.py"
    shutil.copy2(model, run_dir / "custom_model.py")
    env = os.environ.copy()
    env.update({"PYTHONPATH": os.pathsep.join((str(run_dir), str(work / "official/grading-original"))),
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "MPLBACKEND": "Agg",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""})
    command = [sys.executable, str(source), "--seconds", "265", "--router-steps", "2500",
               "--o-steps", "2500", "--validation-per-region", "16384", "--i-safety", "1.5"]
    if not baseline:
        command += ["--i-geometry-bands", str(bands), "--i-geometry-axis", axis]
    report = {"candidate": key, "geometry_bands": bands, "geometry_axis": axis,
              "training_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "model_source_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "configuration_count": len(common.read_config_values(config_dir)),
              "training": common.execute(command, run_dir, env, run_dir / "training.log", 350)}
    if report["training"]["exit_code"] != 0 or not (run_dir / "model.pt").is_file():
        raise RuntimeError(f"Training failed for {run_dir.name}; inspect retained log.")
    return report, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="work/field-round2")
    args = parser.parse_args()
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Official generator requires Linux x86-64 CPython 3.13.")
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    common.BASELINE_REV = BASELINE_REV
    setup_start = time.monotonic()
    configs = common.prepare(work)
    summary = {
        "round": 2, "baseline_commit": BASELINE_REV,
        "official_revision": common.OFFICIAL_REV, "dataset_revision": common.DATA_REV,
        "device": "CPU", "torch_threads": 4,
        "note": "Development re-evaluation: earlier released A/B results motivated this change. Selection uses only public generated holdouts; this is not independent final confirmation or a hidden leaderboard submission.",
        "setup_download_seconds": time.monotonic() - setup_start,
        "selection": {"split": "public/train_config", "seeds": VALIDATION_SEEDS,
                      "points_per_region": VALIDATION_POINTS, "candidates": {}}, "runs": {}}
    def save():
        (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run_start = time.monotonic()
    for candidate in CANDIDATES:
        key = candidate[0]
        run_dir = work / ("selection_" + key)
        report, env = train(work, run_dir, configs["public"], candidate)
        report["validation"] = {str(seed): common.score_run(
            run_dir, configs["public"], env, label=f"validation_{seed}",
            eval_seed=seed, n_per_region=VALIDATION_POINTS) for seed in VALIDATION_SEEDS}
        report["mean_validation_score"] = sum(
            item["result"]["score"]["total_score"] for item in report["validation"].values()) / len(VALIDATION_SEEDS)
        summary["selection"]["candidates"][key] = report
        print("ROUND2_SELECTION", key, json.dumps(report), flush=True)
        save()
    selected = max(CANDIDATES, key=lambda item: (
        summary["selection"]["candidates"][item[0]]["mean_validation_score"], -item[1]))
    summary["selection"]["selected"] = {"candidate": selected[0], "bands": selected[1], "axis": selected[2]}
    print("ROUND2_SELECTED", json.dumps(summary["selection"]["selected"]), flush=True)
    save()
    # Retrain both fixed implementations separately on each mounted configuration.
    for split, config_dir in configs.items():
        summary["runs"][split] = {}
        for role, candidate in (("repository_baseline", CANDIDATES[0]), ("selected_candidate", selected)):
            run_dir = work / (split + "_" + role)
            report, env = train(work, run_dir, config_dir, candidate)
            report.update(common.score_run(run_dir, config_dir, env))
            summary["runs"][split][role] = report
            print("ROUND2_FINAL_RESULT", split, role, json.dumps(report), flush=True)
            save()
    summary["run_seconds_excluding_setup"] = time.monotonic() - run_start
    save()


if __name__ == "__main__":
    main()
