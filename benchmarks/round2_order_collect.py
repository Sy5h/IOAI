"""Collect aggregate Order results and cross-check pinned official scoring.

No model inference is run. Answers, audio and development caches stay in work/.
The published records contain aggregate scores, timing, hashes and clean logs.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

BASELINE_COMMIT = "991a49b48092286967b5bd7cada465e336ebc4af"
OFFICIAL_REVISION = "e1963801aa4cf33a5a03520b39fa6620a7894c54"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--round-dir", required=True, type=Path)
    parser.add_argument("--baseline-source", required=True, type=Path)
    parser.add_argument("--official-scorer", required=True, type=Path)
    parser.add_argument("--workspace-root", default=".", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    replacements = [(str(Path(sys.executable).parent), "<python>"), (str(root), "<workspace>")]
    replacements.sort(key=lambda pair: len(pair[0]), reverse=True)

    def sanitize(value):
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, str):
            for original, replacement in replacements:
                for variant in (original, original.replace("\\", "/")):
                    variants = [variant]
                    for _ in range(3):
                        variants.append(json.dumps(variants[-1])[1:-1])
                    for escaped in reversed(variants):
                        value = value.replace(escaped, replacement)
            value = re.sub(r"(?i)[a-z]:[\\/]+Users[\\/]+[^\\/\r\n\"']+", "<user-home>", value)
        return value

    helper_path = Path(__file__).with_name("audit_order_robot_scores.py")
    spec = importlib.util.spec_from_file_location("official_metric_loader", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    compiled = helper.functions(args.official_scorer, "order")
    baseline_hash = hashlib.sha256(args.baseline_source.read_bytes()).hexdigest()
    paths = [("historical_baseline", args.base_dir / "runs" / f"order_baseline_{split}" / "result.json")
             for split in ("pretest", "a", "b")]
    paths += [("candidate", args.round_dir / "order-runs" / split / "result.json")
              for split in ("pretest-corrected", "leaderboard_a", "leaderboard_b")]
    failure_path = args.round_dir / "order-runs" / "pretest" / "result.json"
    if failure_path.exists():
        paths.append(("failed_dispatch", failure_path))
    runs, audit, logs, pending = [], [], [], []
    for variant, path in paths:
        if not path.exists():
            pending.append(str(path))
            continue
        record = read(path)
        record["variant"] = variant
        record["measurement_round"] = 1 if variant == "historical_baseline" else 2
        if variant == "historical_baseline" and record["script_sha256"] != baseline_hash:
            raise ValueError("Historical baseline source differs from the comparison baseline")
        log = (path.parent / "solver.log").read_text(encoding="utf-8", errors="replace")
        record["ordered_dialogues"] = len(re.findall(r"(?m)^ordered dialogue ", log))
        if record["status"] == "completed" and "result" in record:
            split = record["evaluation_split"]
            if split in {"leaderboard_a", "leaderboard_b"}:
                split = "test_" + split
            truth_file = "pretest_answers.json" if "pretest" in split else split + "_answers.json"
            captured = []

            def fail(message, **kwargs):
                raise ValueError(message)

            namespace = {"json": json, "sys": sys, "SUBMISSION_PATH": path.parent / "answers.json",
                         "HIDDEN_ANSWERS": args.base_dir / "scorer-only" / "order" / truth_file,
                         "fail": fail, "write_report": lambda **values: captured.append(values)}
            exec(compiled, namespace)
            namespace["score_submission"]()
            official = captured[0]["score"]
            local = round(record["result"]["score"], 4)
            checked = {"run": path.parent.name, "variant": variant,
                       "official_score_rounded_4dp": official, "benchmark_score_rounded_4dp": local,
                       "matches": official == local, "details": captured[0]["details"]}
            if not checked["matches"]:
                raise ValueError(checked)
            audit.append(checked)
        else:
            record["score"] = None
        runs.append(record)
        logs.append(f"RUN {variant}/{path.parent.name}\n" + log)
    if pending and not args.allow_pending:
        raise FileNotFoundError("Pending results: " + ", ".join(pending))
    result = {
        "measurement_date": "2026-10-16", "task": "Find the Order",
        "baseline_commit": BASELINE_COMMIT, "baseline_source_sha256": baseline_hash,
        "data_revision": "71a35b5a566665e242416eba45f018dc378663b8",
        "model_revisions": {"whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
                            "Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987"},
        "selection": read(args.round_dir / "order-selection.json"),
        "development_search": [{k: v for k, v in row.items() if k != "per_dialogue"}
                               for row in read(args.round_dir / "order-search.json")],
        "notes": [
            "The only solver change is the default beam width from 16 to 256; models, ASR, Qwen scoring, speaker filter and time budget are unchanged.",
            "All eight alternatives were selected on public/pretrain 20 dialogues. first10/last10 are descriptive subsets of the same development sample, not untouched holdouts.",
            "Development reused Qwen/ASR scores, with LM batch8 and all eligible edges. Search-only seconds exclude model work and are not final runtime. The cache is not supplied to final solvers, which keep original LM batch4.",
            "Historical baseline source SHA256 is identical to the 991a49b snapshot. Historical timings are earlier runs, not fresh paired round-two baseline measurements; small timing changes cannot be attributed to beam width.",
            "Each recorded successful run used a fresh solver process and recomputed transcription/model scores. Time includes imports, local model/data loading, inference, answer writing and exit; downloads and scoring are excluded.",
            "Prior A/B were observed before this development round. The beam was fixed before the new pretest/A/B evaluations. These are released-data development re-evaluations, not independent blind tests.",
            "GPU was scheduled per task, but CPU, I/O and other agents' work could overlap. Driver/OS caches may be warm. The original contest Linux/Podman environment was not reproduced.",
            "An initial pretest dispatch used a nonexistent input folder and failed before inference. It has no score. The corrected path was rerun; both records are preserved.",
            "No raw transcripts, answer arrays, audio, cache contents or model weights are published. Scorer-only reference files are read only after solver execution.",
        ],
        "official_scoring_crosscheck": {"revision": OFFICIAL_REVISION,
            "source_sha256": helper.SOURCES["order"][1],
            "url": f"https://github.com/IOAI-official/IOAI-2026/blob/{OFFICIAL_REVISION}/Individual-Contest/1_Find_the_Order/code/grading-original/check.py",
            "scope": "Pinned validate_permutation, score_one, score_submission functions only; no container or resource wrapper.",
            "all_match": all(row["matches"] for row in audit), "runs": audit},
        "runs": runs, "pending": pending,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sanitize(result), indent=2, ensure_ascii=False), encoding="utf-8")
    args.output.with_name("round2_order_runs.log").write_text(sanitize("\n".join(logs)), encoding="utf-8")
    print(f"Collected {len(runs)} records; official score matches {len(audit)}/{len(audit)}; pending {len(pending)}")


if __name__ == "__main__":
    main()
