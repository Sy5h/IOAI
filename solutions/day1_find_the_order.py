"""IOAI 2026 Day 1: Find the Order. Proposed offline, budgeted solution.

No pretrained weights or test answers are embedded. Run with --help for options.
Only the Python standard library is imported until model inference is requested.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import time


def order_to_ranks(order):
    """Invert a playback permutation: answer[chunk_id] = chronological rank."""
    n = len(order)
    if sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation of 0..n-1")
    ranks = [0] * n
    for position, chunk in enumerate(order):
        ranks[chunk] = position
    return ranks


def prefix_order(n, prefix):
    prefix = list(prefix)
    if len(prefix) != min(2, n) or len(set(prefix)) != len(prefix):
        raise ValueError("prefix must contain the first two distinct chunk IDs")
    if any(type(x) is not int or x < 0 or x >= n for x in prefix):
        raise ValueError("prefix contains an invalid chunk ID")
    return prefix + [i for i in range(n) if i not in prefix]


def beam_order(edges, prefix, width=16, speakers=None):
    """Search an adjacency objective; prefix is fixed and every ID appears once.

    This objective is a heuristic, not the contest's pairwise-order metric.
    Speaker labels, when supplied, must agree with strict A/B alternation.
    """
    n = len(edges)
    base = prefix_order(n, prefix)
    if width < 1:
        raise ValueError("beam width must be positive")
    if n <= len(prefix):
        return base
    if speakers is not None:
        if len(speakers) != n or speakers[prefix[0]] == speakers[prefix[1]]:
            speakers = None
        else:
            first = speakers[prefix[0]]
            if sum(s == first for s in speakers) != (n + 1) // 2:
                speakers = None
    beam = [(0.0, tuple(prefix), sum(1 << i for i in prefix))]
    for position in range(len(prefix), n):
        expanded = []
        for score, path, used in beam:
            for j in range(n):
                if used & (1 << j):
                    continue
                if speakers is not None:
                    expected = speakers[prefix[position % 2]]
                    if speakers[j] != expected:
                        continue
                edge = float(edges[path[-1]][j])
                if not math.isfinite(edge):
                    edge = -1e6
                expanded.append((score + edge, path + (j,), used | (1 << j)))
        if not expanded:
            return beam_order(edges, prefix, width=width, speakers=None)
        expanded.sort(key=lambda item: (-item[0], item[1]))
        beam = expanded[:width]
    return list(beam[0][1])


def load_dialogues(test_dir):
    test_dir = Path(test_dir)
    prefixes = json.loads((test_dir / "prefix.json").read_text(encoding="utf-8"))
    records = []
    for folder in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        chunks = {}
        for path in folder.iterdir():
            match = re.fullmatch(r"chunk_(\d+)\.wav", path.name)
            if match:
                chunks[int(match.group(1))] = path
        if not chunks:
            continue
        n = len(chunks)
        if sorted(chunks) != list(range(n)):
            raise ValueError(f"non-contiguous chunk IDs in {folder}")
        prefix = prefixes.get(folder.name)
        if prefix is None:
            raise ValueError(f"missing prefix for dialogue {folder.name}")
        order = prefix_order(n, prefix)
        records.append({"id": folder.name, "paths": [chunks[i] for i in range(n)],
                        "prefix": list(prefix), "order": order})
    if not records:
        raise ValueError(f"no dialogue audio found in {test_dir}")
    return records


def write_answers(records, output):
    answers = {r["id"]: order_to_ranks(r["order"]) for r in records}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(answers), encoding="utf-8")
    return answers


def read_audio(path):
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, audio = wavfile.read(str(path))
    dtype = audio.dtype
    audio = audio.astype(np.float32)
    if dtype.kind in "iu":
        if dtype.kind == "u":
            midpoint = float(2 ** (np.iinfo(dtype).bits - 1))
            audio = (audio - midpoint) / midpoint
        else:
            audio /= float(2 ** (np.iinfo(dtype).bits - 1))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(int(rate), 16000)
        audio = resample_poly(audio, 16000 // divisor, int(rate) // divisor)
    if not audio.size:
        audio = np.zeros(400, dtype=np.float32)
    return np.asarray(audio, dtype=np.float32)


def acoustic_features(audio):
    """Small MFCC summary, using scipy rather than a speaker-recognition model."""
    import numpy as np
    from scipy.fft import dct
    from scipy.signal import stft

    audio = np.pad(audio, (0, max(0, 400 - len(audio))))
    _, _, spectrum = stft(audio, fs=16000, nperseg=400, noverlap=240,
                          nfft=512, boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    frequencies = np.linspace(0, 8000, power.shape[0])
    mel_max = 2595 * np.log10(1 + 8000 / 700)
    hz = 700 * (10 ** (np.linspace(0, mel_max, 28) / 2595) - 1)
    bank = []
    for lo, center, hi in zip(hz[:-2], hz[1:-1], hz[2:]):
        bank.append(np.maximum(0, np.minimum((frequencies - lo) / (center - lo),
                                              (hi - frequencies) / (hi - center))))
    mel = np.asarray(bank) @ power
    energy = power.sum(axis=0)
    voiced = energy > max(float(energy.max()) * 0.01, 1e-12)
    if not voiced.any():
        voiced[:] = True
    coeffs = dct(np.log(mel[:, voiced] + 1e-10), axis=0, norm="ortho")[:13]
    return np.concatenate((coeffs.mean(axis=1), coeffs.std(axis=1)))


def reliable_speakers(features, prefix):
    """Return binary labels only for a clean, correctly balanced two-cluster split.

    This conservative check reduces damage from clustering by words/noise instead
    of voice. It is still a heuristic; --no-speakers disables the constraint.
    """
    import numpy as np

    x = np.asarray(features, dtype=np.float64)
    n = len(x)
    if n < 3 or len(prefix) != 2 or not np.isfinite(x).all():
        return None
    scale = x.std(axis=0)
    active = scale > 1e-6
    if not active.any():
        return None
    x = (x[:, active] - x[:, active].mean(axis=0)) / scale[active]
    centers = x[list(prefix)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(12):
        distances = ((x[:, None] - centers[None]) ** 2).mean(axis=2)
        labels = distances.argmin(axis=1)
        labels[prefix[0]], labels[prefix[1]] = 0, 1
        if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
            return None
        centers = np.stack([x[labels == k].mean(axis=0) for k in range(2)])
    distances = ((x[:, None] - centers[None]) ** 2).mean(axis=2)
    if distances[prefix[0]].argmin() != 0 or distances[prefix[1]].argmin() != 1:
        return None
    confidence = np.abs(distances[:, 0] - distances[:, 1]) / (distances.sum(axis=1) + 1e-9)
    if (labels == 0).sum() != (n + 1) // 2 or float(confidence.min()) < 0.20:
        return None
    return labels.tolist()


def transcribe(records, args, stop_at):
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.whisper, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.whisper, local_files_only=True, torch_dtype=dtype).to(device).eval()
    # Complete short dialogues first: each dialogue has equal scoring weight.
    for record in sorted(records, key=lambda r: len(r["paths"])):
        if time.monotonic() >= stop_at:
            break
        texts, features = [], []
        for start in range(0, len(record["paths"]), args.asr_batch):
            if time.monotonic() >= stop_at:
                break
            arrays = [read_audio(p) for p in record["paths"][start:start + args.asr_batch]]
            if not args.no_speakers:
                features.extend(acoustic_features(a) for a in arrays)
            # A single turn normally fits the Whisper window. This explicit cap
            # bounds generation time; very long turns may lose their endings.
            arrays = [a[:int(args.max_audio_seconds * 16000)] for a in arrays]
            batch = processor(arrays, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            with torch.inference_mode():
                generated = model.generate(
                    batch.input_features.to(device=device, dtype=dtype),
                    attention_mask=batch.attention_mask.to(device),
                    language="en", task="transcribe", do_sample=False, num_beams=1,
                    max_new_tokens=args.asr_tokens)
            texts.extend(t.strip() for t in processor.batch_decode(generated, skip_special_tokens=True))
        if len(texts) == len(record["paths"]):
            record["texts"] = texts
            if features:
                record["speakers"] = reliable_speakers(np.asarray(features), record["prefix"])
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def conditional_scores(model, tokenizer, pairs, batch_size, max_context, max_target, stop_at):
    """Mean target-token log probabilities; padding/context tokens are not scored."""
    import torch

    scores = []
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad nor EOS token")
    for start in range(0, len(pairs), batch_size):
        if time.monotonic() >= stop_at:
            return None
        batch = []
        for context, target in pairs[start:start + batch_size]:
            context_ids = tokenizer.encode(context, add_special_tokens=False)[-max_context:]
            target_ids = tokenizer.encode(" " + (target.strip() or "..."), add_special_tokens=False)[:max_target]
            if not context_ids:
                context_ids = [pad_id]
            if not target_ids:
                target_ids = [pad_id]
            batch.append((context_ids + target_ids, len(context_ids)))
        length = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), length), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        target_mask = torch.zeros((len(batch), length - 1), dtype=torch.bool, device=device)
        for row, (tokens, context_length) in enumerate(batch):
            ids[row, :len(tokens)] = torch.tensor(tokens, device=device)
            mask[row, :len(tokens)] = 1
            target_mask[row, context_length - 1:len(tokens) - 1] = True
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
            # Select only target positions before the large-vocabulary softmax.
            losses = torch.nn.functional.cross_entropy(
                logits[target_mask].float(), ids[:, 1:][target_mask], reduction="none")
            counts = target_mask.sum(dim=1).tolist()
            offset = 0
            for count in counts:
                scores.append(-float(losses[offset:offset + count].mean().item()))
                offset += count
        del logits, losses
    return scores


def language_edges(model, tokenizer, texts, args, stop_at, prefix=None, speakers=None):
    fixed = set(prefix or [])
    targets = [j for j in range(len(texts)) if j not in fixed]
    baseline = conditional_scores(model, tokenizer, [("Speaker B:", texts[j]) for j in targets],
                                  args.lm_batch, args.context_tokens, args.target_tokens, stop_at)
    if baseline is None:
        return None
    baseline = dict(zip(targets, baseline))
    # Never score edges that the fixed prefix or reliable speaker alternation
    # makes unreachable. This saves model work without changing the search.
    sources = [i for i in range(len(texts)) if not prefix or i != prefix[0]]
    indices = [(i, j) for i in sources for j in targets
               if i != j and (speakers is None or speakers[i] != speakers[j])]
    pairs = [("Speaker A: " + texts[i] + "\nSpeaker B:", texts[j]) for i, j in indices]
    conditioned = conditional_scores(model, tokenizer, pairs, args.lm_batch,
                                     args.context_tokens, args.target_tokens, stop_at)
    if conditioned is None:
        return None
    edges = [[0.0] * len(texts) for _ in texts]
    for (i, j), score in zip(indices, conditioned):
        # This reweighting affects beam pruning. For complete orders, the sum of
        # baseline[j] is constant because every non-prefix target appears once.
        edges[i][j] = score - baseline[j]
    return edges


def reorder_with_qwen(records, args, stop_at):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    candidates = [r for r in records if "texts" in r]
    if not candidates or time.monotonic() >= stop_at - 10:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.qwen, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen, local_files_only=True, torch_dtype=dtype).to(device).eval()
    for record in sorted(candidates, key=lambda r: len(r["paths"])):
        if time.monotonic() >= stop_at:
            break
        edges = language_edges(model, tokenizer, record["texts"], args, stop_at,
                               record["prefix"], record.get("speakers"))
        if edges is None:
            break
        record["order"] = beam_order(edges, record["prefix"], args.beam,
                                     record.get("speakers"))
        write_answers(records, args.output)
        print(f"ordered dialogue {record['id']} ({len(record['paths'])} turns)", flush=True)
    del model, tokenizer


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default="dataset/test_public")
    parser.add_argument("--output", default="answers.json")
    parser.add_argument("--whisper", default="models/whisper-small")
    parser.add_argument("--qwen", default="models/qwen2.5-0.5b")
    parser.add_argument("--budget", type=float, default=540, help="soft total seconds, below statement's 600s")
    parser.add_argument("--asr-fraction", type=float, default=0.55)
    parser.add_argument("--asr-batch", type=int, default=8)
    parser.add_argument("--lm-batch", type=int, default=4)
    parser.add_argument("--asr-tokens", type=int, default=128)
    parser.add_argument("--max-audio-seconds", type=float, default=30)
    parser.add_argument("--context-tokens", type=int, default=96)
    parser.add_argument("--target-tokens", type=int, default=64)
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--prefix-only", action="store_true", help="stdlib-only baseline; no model loading")
    parser.add_argument("--no-speakers", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if min(args.asr_batch, args.lm_batch, args.asr_tokens, args.context_tokens,
           args.target_tokens, args.beam) < 1 or not 0 < args.max_audio_seconds <= 30:
        raise ValueError("batch sizes, token limits and beam must be positive; audio window must be in (0, 30]")
    if not 0 < args.asr_fraction < 1:
        raise ValueError("--asr-fraction must be between 0 and 1")
    started = time.monotonic()
    records = load_dialogues(args.test_dir)
    answers = write_answers(records, args.output)
    if args.prefix_only or args.budget <= 20:
        print(f"wrote prefix baseline for {len(answers)} dialogues")
        return answers
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    deadline = started + args.budget - 15
    try:
        transcribe(records, args, min(deadline, started + args.budget * args.asr_fraction))
    except Exception as exc:
        print(f"ASR stopped; completed dialogues/fallback retained: {exc}", file=sys.stderr)
        gc.collect()
    try:
        reorder_with_qwen(records, args, deadline)
    except Exception as exc:
        print(f"Qwen stopped; completed dialogues/fallback retained: {exc}", file=sys.stderr)
    answers = write_answers(records, args.output)
    print(f"wrote {len(answers)} valid permutations in {time.monotonic() - started:.1f}s")
    return answers


if __name__ == "__main__":
    main()
