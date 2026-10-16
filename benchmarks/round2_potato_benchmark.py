"""Run fresh baseline/current processes against released Potato A/B judges.

Only benchmark_potato (the parent judge) receives --evaluation-dir. The solver
receives public resources and normal comparison feedback, never target words or
private embeddings. Keep raw run files under the Git-ignored work directory.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

BASELINE_COMMIT = "991a49b48092286967b5bd7cada465e336ebc4af"
ROOT = Path(__file__).resolve().parent


def one_run():
    spec = importlib.util.spec_from_file_location("round2_original_potato_harness", ROOT / "benchmark_potato.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.BASELINE_COMMIT = BASELINE_COMMIT
    sys.argv.pop(1)
    harness.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output.parent / "selection_frozen.json"
    if not selection_path.exists():
        development_path = output.parent / "development_64.json"
        confirmation_path = output.parent / "confirmation_128.json"
        selection = {"baseline_commit": BASELINE_COMMIT,
                     "candidate_sha256": hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
                     "fixed_before_this_round_AB": True,
                     "selection": "Re-evaluate the supplied fixed candidate; development provenance recorded below."}
        if development_path.exists() and confirmation_path.exists():
            development = json.loads(development_path.read_text(encoding="utf-8"))
            confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
            selection["selected"] = development["selected"]
            selection["confirmation_baseline"] = confirmation["candidates"]["baseline"]["robust_mean"]
            selection["confirmation_candidate"] = confirmation["candidates"][development["selected"]]["robust_mean"]
        else:
            selection["development_provenance"] = "Development files absent; this is a fixed-code reproduction, not a new selection."
        selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    for split in ("leaderboard_a", "leaderboard_b"):
        for label, source in (("baseline", args.baseline), ("candidate", args.candidate)):
            tag = f"{label}_{split}"
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--one-run",
                       "--script", str(Path(source).resolve()),
                       "--data-dir", str(Path(args.data_dir).resolve()),
                       "--evaluation-dir", str(Path(args.evaluation_dir).resolve()),
                       "--suite", split, "--output-dir", str(output.resolve()), "--tag", tag]
            with (output / f"{tag}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            path = output / f"{tag}.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                print(json.dumps({k: record[k] for k in ("suite", "score", "wins", "run_seconds", "status")}
                                 | {"variant": label}), flush=True)
            else:
                print(json.dumps({"tag": tag, "exit_code": result.returncode, "status": "failed"}), flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one-run":
        one_run()
    else:
        main()
