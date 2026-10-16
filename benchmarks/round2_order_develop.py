"""Reproduce the eight PUBLIC/pretrain-only sequence-search alternatives.

cache uses the permitted local Whisper/Qwen models; search is CPU-only. Cache
files belong under work/ and are never submitted or supplied to final solvers.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import time

BASELINE_COMMIT = "991a49b48092286967b5bd7cada465e336ebc4af"
CONFIGS = [(16, True, 0, False), (64, True, 0, False), (256, True, 0, False),
           (64, True, 0, True), (64, True, .25, True), (64, True, 1., True),
           (64, False, 0, False), (64, False, .25, True)]


def load_source(path):
    spec = importlib.util.spec_from_file_location("order_development_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    solver = load_source(args.source)
    parameters = solver.make_parser().parse_args([])
    parameters.whisper = str(Path(args.models_dir) / "whisper-small")
    parameters.qwen = str(Path(args.models_dir) / "qwen2.5-0.5b")
    parameters.lm_batch = 8
    records = solver.load_dialogues(args.data_dir)
    if len(records) != 20:
        raise ValueError("This experiment expects the 20-dialogue public/pretrain split")
    started = time.monotonic()
    solver.transcribe(records, parameters, started + 600)
    if any("texts" not in r for r in records):
        raise RuntimeError("Development transcription did not complete")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(parameters.qwen, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(parameters.qwen, local_files_only=True,
                                               torch_dtype=dtype).to(device).eval()
    cached = []
    for row in records:
        edges = solver.language_edges(model, tokenizer, row["texts"], parameters,
                                      time.monotonic() + 300, row["prefix"], None)
        if edges is None:
            raise RuntimeError("Development edge scoring did not complete")
        # Search needs no raw transcript or audio in its cache.
        cached.append({"id": row["id"], "prefix": row["prefix"],
                       "speakers": row.get("speakers"), "edges": edges})
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"split": "public/pretrain", "baseline_commit": BASELINE_COMMIT,
                 "elapsed_seconds": time.monotonic() - started, "records": cached}), encoding="utf-8")
        print(f"Cached {len(cached)}/20 development dialogues", flush=True)


def search_order(solver, edges, prefix, width, speakers, global_weight=0., center=False):
    import numpy as np
    values = np.array(edges)
    count = len(values)
    if center:
        for target in range(count):
            if target not in prefix:
                valid = [source for source in range(count) if source != target and source != prefix[0]
                         and (speakers is None or speakers[source] != speakers[target])]
                values[:, target] -= float(np.mean(values[valid, target]))
    if global_weight == 0:
        return solver.beam_order(values, prefix, width, speakers)
    fixed = set(prefix)
    values = values.tolist()
    beam = [(0., tuple(prefix), sum(1 << value for value in prefix))]
    for position in range(2, count):
        expanded = []
        for score, path, used in beam:
            for target in range(count):
                if used & (1 << target):
                    continue
                if speakers is not None and speakers[target] != speakers[prefix[position % 2]]:
                    continue
                extra = sum(values[source][target] - values[target][source]
                            for source in path if source not in fixed) / (count - 2)
                expanded.append((score + values[path[-1]][target] + global_weight * extra,
                                 path + (target,), used | (1 << target)))
        expanded.sort(key=lambda item: (-item[0], item[1]))
        beam = expanded[:width]
    return list(beam[0][1])


def search(args):
    import numpy as np
    solver = load_source(args.source)
    payload = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    records = payload["records"]
    if payload["split"] != "public/pretrain" or len(records) != 20:
        raise ValueError("Search selection must use the complete public/pretrain cache")
    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    if set(truth) != {r["id"] for r in records}:
        raise ValueError("Development truth IDs and cached IDs differ")
    results = []
    for width, use_speakers, weight, center in CONFIGS:
        started = time.perf_counter()
        scores = []
        for row in records:
            order = search_order(solver, row["edges"], row["prefix"], width,
                                 row["speakers"] if use_speakers else None, weight, center)
            ranks, target = solver.order_to_ranks(order), truth[row["id"]]
            count = len(ranks)
            correct = sum((ranks[i] > ranks[j]) == (target[i] > target[j])
                          for i in range(count) for j in range(i + 1, count))
            scores.append(100 * correct / (count * (count - 1) / 2))
        record = {"width": width, "speakers": use_speakers, "global_weight": weight, "center": center,
                  "mean": float(np.mean(scores)), "first10": float(np.mean(scores[:10])),
                  "last10": float(np.mean(scores[10:])), "seconds": time.perf_counter() - started,
                  "per_dialogue": scores}
        results.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "per_dialogue"}), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.selection_output:
        selected = max(results, key=lambda row: row["mean"])
        selection = {"baseline_commit": BASELINE_COMMIT, "development_split": "public/pretrain:20 dialogues",
                     "created_utc": datetime.now(timezone.utc).isoformat(),
                     "selected_change": {k: selected[k] for k in ("width", "speakers", "global_weight", "center")},
                     "reason": "Largest mean of the eight fixed development alternatives; no A/B inputs or labels read.",
                     "cache_note": "LM batch8 development cache; final solver uses original LM batch4 and no cache.",
                     "protocol": "Freeze this record before new final evaluations; historical A/B have already been observed."}
        Path(args.selection_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.selection_output).write_text(json.dumps(selection, indent=2), encoding="utf-8")


def main():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--source", required=True)
    c.add_argument("--data-dir", required=True)
    c.add_argument("--models-dir", required=True)
    c.add_argument("--output", required=True)
    s = sub.add_parser("search")
    s.add_argument("--source", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--truth", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--selection-output", help="Write the development decision before final evaluation")
    args = parser.parse_args()
    (cache if args.operation == "cache" else search)(args)


if __name__ == "__main__":
    main()
