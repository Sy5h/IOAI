"""Publish aggregate Potato round-two measurements without answers or local paths."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    work = Path(args.work_dir)
    workspace = Path(args.workspace_root).resolve()
    root_variants = {str(workspace), str(workspace).replace("\\", "/")}
    def sanitize(value):
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, str):
            value = value.replace(json.dumps(sys.executable)[1:-1], "python")
            value = value.replace(sys.executable, "python")
            for root in sorted(root_variants, key=len, reverse=True):
                value = value.replace(json.dumps(root)[1:-1], "<workspace>")
                value = value.replace(root, "<workspace>")
            return value
        return value
    def read(relative):
        return json.loads((work / relative).read_text(encoding="utf-8"))
    result = {
        "measurement_date": "2026-10-16",
        "task": "Potato",
        "baseline_commit": "991a49b48092286967b5bd7cada465e336ebc4af",
        "data_revision": "122716fc56a212e58b1756158bdeb4c2ea2da5e6",
        "metric": "100 * mean(0 if not solved, else 1 - .02 * max(0, winning_turn - 10))",
        "selection": read("selection_frozen.json"),
        "development": read("development_64.json"),
        "confirmation": read("confirmation_128.json"),
        "notes": [
            "No extra model, training data, GPU, or private-space fitting. Only the supplied public embedding matrix is used by the solver.",
            "Twelve policies were compared on 64 non-public-target vocabulary words using three synthetic perturbed public-space judges. Selection used the mean noisy-judge score, not the exact public-space score.",
            "The selected policy was checked once on 128 disjoint words and new random projections. Its mean gain was small and the strongest distortion worsened; these synthetic scores are not official scores.",
            "The candidate was frozen before this round's A/B. Earlier A/B performance was already known, so these released A/B runs are development re-evaluations, not independent blind confirmation. No further tuning followed the results.",
            "All A/B targets and private embeddings remain scorer-only. Solver stdin contains only normal comparison feedback. This is process separation, not a security sandbox.",
            "Each final row is one fresh solver process: imports, public resource loading, all 120 games, writing protocol outputs, and process exit are included. Judge/data preparation and prior downloads are excluded.",
            "CPU only, BLAS 4 threads. Other agents' CPU/GPU/file work may run concurrently; OS file caches may be warm. Times are not dedicated-hardware throughput estimates.",
            "All four final runs completed. Raw per-game records and original logs remain in the Git-ignored work directory; published records omit target hashes and per-game answers.",
        ],
        "adopted_parameters": {"temperature": .06, "component_temperatures": [.03, .06, .12],
                               "error_floor": .12, "posterior_mode": "mixture", "guess_after": 8,
                               "support_size": 256},
        "runs": [],
    }
    logs = []
    for split in ("leaderboard_a", "leaderboard_b"):
        for label in ("baseline", "candidate"):
            name = f"{label}_{split}"
            run = read(f"evaluation/{name}.json")
            run.pop("records", None)
            run["variant"] = label
            result["runs"].append(run)
            logs.append(f"RUN {name}\n" + (work / "evaluation" / f"{name}.log").read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sanitize(result), indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_name("round2_potato_runs.log").write_text(sanitize("\n".join(logs)), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
