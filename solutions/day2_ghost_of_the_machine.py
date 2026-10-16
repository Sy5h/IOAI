"""Ghost of the Machine: frozen BGE features + logistic regression + one switch.

This module is self-contained and can be inserted into one solution.ipynb cell.
Models are loaded only from a local directory. No pretrained weights other than
the allowed BGE-base-en-v1.5 are used. No contest-score claim is made for this code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np


class BudgetExceeded(RuntimeError):
    pass


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_answers(data_dir):
    data_dir = Path(data_dir)
    candidates = [data_dir / "answers.jsonl", data_dir.parent / f"{data_dir.name}_answers.jsonl"]
    for path in candidates:
        if path.is_file():
            return {r["id"]: int(r["boundary_char_index"]) for r in read_jsonl(path)}
    raise FileNotFoundError(f"No answers.jsonl or {data_dir.name}_answers.jsonl for {data_dir}")


def boundary_score(predictions, answers):
    if not answers:
        return 0.0
    return 100.0 * float(np.mean([
        math.exp(-abs(int(predictions[key]) - value) / 100.0)
        for key, value in answers.items()
    ]))


def learn_separators(rows, answers, min_coverage=0.995):
    """Infer boundary suffixes from TRAIN labels, including whitespace offsets.

    Prefer a punctuation/newline suffix that covers most labelled boundaries.
    When no single suffix does, use a small greedy cover. An empty result means
    that the general sentence fallback must be used. No separator is assumed true.
    """
    coverage = {}
    total = len(rows)
    for i, row in enumerate(rows):
        boundary = answers[row["id"]]
        before = row["text"][:boundary]
        for size in range(1, min(8, len(before)) + 1):
            suffix = before[-size:]
            if (re.fullmatch(r"[\s.!?…\"'“”‘’]+", suffix)
                    and (any(c in suffix for c in ".!?…") or "\n" in suffix)):
                coverage.setdefault(suffix, set()).add(i)
    if not coverage:
        return []
    frequent = [s for s, ids in coverage.items() if len(ids) / max(total, 1) >= min_coverage]
    if frequent:
        # Long suffixes such as '. ' exclude abbreviation periods and preserve
        # the exact convention of the supplied character labels.
        chosen = min(frequent, key=lambda s: (sum(r["text"].count(s) for r in rows), -len(s)))
        return [chosen]
    remaining = set(range(total))
    selected = []
    while remaining and len(selected) < 4:
        candidate = max(coverage, key=lambda s: (len(coverage[s] & remaining), len(s)))
        gain = coverage[candidate] & remaining
        if not gain or len(gain) < max(2, 0.01 * total):
            break
        selected.append(candidate)
        remaining -= gain
        if len(remaining) <= 0.02 * total:
            break
    return selected


def sentence_spans(text, separators):
    """Return nonempty (start, end) spans without changing the original text."""
    positions = {0, len(text)}
    for separator in separators:
        offset = text.find(separator)
        while offset >= 0:
            positions.add(offset + len(separator))
            offset = text.find(separator, offset + 1)
    if not separators or len(positions) <= 2:
        positions.update(m.end() for m in re.finditer(r"[.!?][\"'”’]*\s+|\n\s*\n", text))
    ordered = sorted(positions)
    spans = [(start, end) for start, end in zip(ordered, ordered[1:])
             if text[start:end].strip()]
    return spans or [(0, len(text))]


def one_switch_logits(machine_probabilities):
    """Return log likelihoods for boundaries before sentence k, k = 0..n."""
    p = np.clip(np.asarray(machine_probabilities, dtype=float), 1e-6, 1 - 1e-6)
    human_prefix = np.concatenate([[0.0], np.cumsum(np.log1p(-p))])
    machine_prefix = np.concatenate([[0.0], np.cumsum(np.log(p))])
    return human_prefix + machine_prefix[-1] - machine_prefix


def fallback_boundary(text, mean_fraction=0.6, min_side_words=180,
                      preferred_positions=None):
    """Snap the mean-position estimate to a legal ORIGINAL-text boundary.

    Sentence detection can return only [0, len(text)] for a valid long passage.
    In that case use word boundaries, not an impossible all-human/all-machine
    output. When the passage is too short for the minimum, keep both sides
    nonempty where possible instead of pretending the requirement is feasible.
    """
    words = list(re.finditer(r"\S+", text))
    guess = float(np.clip(mean_fraction, 0.0, 1.0)) * len(text)
    if len(words) < 2:
        return int(np.clip(round(guess), 1, len(text) - 1)) if len(text) > 1 else 0
    required = max(1, min_side_words)
    if len(words) < 2 * required:
        required = 1
    word_ends = np.asarray([word.end() for word in words], dtype=int)
    if preferred_positions is not None:
        positions = np.asarray(list(preferred_positions), dtype=int)
        left_counts = np.searchsorted(word_ends, positions, side="right")
        legal = ((positions > 0) & (positions < len(text))
                 & (left_counts >= required)
                 & (len(words) - left_counts >= required))
        if legal.any():
            candidates = positions[legal]
            return int(candidates[np.argmin(np.abs(candidates - guess))])
    candidates = np.asarray([words[i].start()
                             for i in range(required, len(words) - required + 1)])
    return int(candidates[np.argmin(np.abs(candidates - guess))])


def choose_boundary(text, spans, probabilities, mean_fraction=0.6,
                    prior_strength=0.25, min_side_words=180,
                    expected_score=False):
    positions = np.asarray([start for start, _ in spans] + [len(text)], dtype=int)
    scores = one_switch_logits(probabilities)
    fractions = positions / max(len(text), 1)
    scores -= prior_strength * ((fractions - mean_fraction) / 0.15) ** 2
    word_ends = np.asarray([m.end() for m in re.finditer(r"\S+", text)])
    left_counts = np.searchsorted(word_ends, positions, side="right")
    legal = ((positions > 0) & (positions < len(text))
             & (left_counts >= min_side_words)
             & (len(word_ends) - left_counts >= min_side_words))
    # A sentence separator is not guaranteed to exist inside the legal region.
    if not legal.any():
        return fallback_boundary(text, mean_fraction, min_side_words)
    candidates, valid_scores = positions[legal], scores[legal]
    if expected_score:
        mass = np.exp(valid_scores - valid_scores.max())
        mass /= mass.sum()
        reward = np.exp(-np.abs(candidates[:, None] - candidates[None, :]) / 100.0)
        index = int(np.argmax(reward @ mass))
    else:
        index = int(np.argmax(valid_scores))
    return int(candidates[index])


def article_split(rows, validation_fraction=0.15, seed=42):
    if len(rows) < 4:
        return list(range(len(rows))), []
    order = np.random.default_rng(seed).permutation(len(rows))
    count = max(1, min(len(rows) - 2, round(len(rows) * validation_fraction)))
    return sorted(order[count:].tolist()), sorted(order[:count].tolist())


def build_sentence_data(rows, separators, answers=None):
    sentences, labels, article_indices, spans_by_article = [], [], [], []
    for article, row in enumerate(rows):
        spans = sentence_spans(row["text"], separators)
        spans_by_article.append(spans)
        boundary = None if answers is None else answers[row["id"]]
        for start, end in spans:
            sentences.append(row["text"][start:end])
            article_indices.append(article)
            # -1 means a span straddles the label; do not teach a false binary label.
            labels.append(-1 if boundary is None or start < boundary < end
                          else int(start >= boundary))
    return sentences, np.asarray(labels), np.asarray(article_indices), spans_by_article


def embed_sentences(sentences, model_dir, batch_size=64, max_length=128,
                    device=None, deadline=None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Allowed local BGE model directory not found: {model_dir}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).eval().to(device)
    if str(device).startswith("cuda"):
        model.half()
    vectors = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(sentences), batch_size):
                if deadline is not None and time.monotonic() > deadline:
                    raise BudgetExceeded("Embedding time budget exhausted")
                tokens = tokenizer(sentences[offset:offset+batch_size], padding=True,
                                   truncation=True, max_length=max_length,
                                   return_tensors="pt").to(device)
                hidden = model(**tokens).last_hidden_state[:, 0].float()
                vectors.append(torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy())
    finally:
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return np.concatenate(vectors, axis=0)


def decode_articles(rows, spans_by_article, article_indices, probabilities,
                    mean_fraction, args):
    return {
        row["id"]: choose_boundary(
            row["text"], spans_by_article[i], probabilities[article_indices == i],
            mean_fraction, args.prior_strength, args.min_side_words, args.expected_score,
        )
        for i, row in enumerate(rows)
    }


def write_predictions(path, rows, predictions):
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(predictions):
        raise ValueError("Exactly one prediction per test ID is required")
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            value = predictions[row["id"]]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError("Boundary must be an integer")
            if not 0 <= int(value) <= len(row["text"]):
                raise ValueError("Boundary is outside the original text")
            f.write(json.dumps({"id": row["id"], "boundary_char_index": int(value)}) + "\n")


def resolve_model_path(path=None):
    if path:
        return path
    root = Path(os.environ.get("MODELS_DIR", "models"))
    for candidate in [root / "bge-base-en-v1.5", root / "BAAI" / "bge-base-en-v1.5"]:
        if candidate.is_dir():
            return str(candidate)
    return str(root / "bge-base-en-v1.5")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test_public")
    parser.add_argument("--model-dir")
    parser.add_argument("--output", default="answers.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device")
    parser.add_argument("--budget-seconds", type=float, default=540.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regularization-c", type=float, default=2.0)
    parser.add_argument("--prior-strength", type=float, default=0.25)
    parser.add_argument("--min-side-words", type=int, default=180)
    parser.add_argument("--expected-score", action="store_true")
    # A notebook kernel supplies its own -f connection-file argument.
    if argv is None and "__file__" not in globals():
        argv = []
    args = parser.parse_args(argv)
    start = time.monotonic()
    train_rows = read_jsonl(Path(args.train_dir) / "data.jsonl")
    test_rows = read_jsonl(Path(args.test_dir) / "data.jsonl")
    answers = read_answers(args.train_dir)
    if not train_rows or not test_rows:
        raise ValueError("Training and test passages must be nonempty")
    train_idx, validation_idx = article_split(train_rows, args.validation_fraction, args.seed)
    fitting_rows = [train_rows[i] for i in train_idx]
    # Separator discovery is also fitted without validation articles.
    separators = learn_separators(fitting_rows, answers)
    mean_fraction = float(np.mean([answers[r["id"]] / max(1, len(r["text"]))
                                  for r in fitting_rows]))
    print(f"Learned separator(s): {separators!r}; article validation={len(validation_idx)}", file=sys.stderr)
    train_text, y, train_article, train_spans = build_sentence_data(train_rows, separators, answers)
    test_text, _, test_article, test_spans = build_sentence_data(test_rows, separators)
    # A usable fallback exists before expensive model work. It is not a claim
    # that the average-position baseline is strong.
    fallback = {r["id"]: fallback_boundary(
                    r["text"], mean_fraction, args.min_side_words,
                    preferred_positions=(s for s, _ in test_spans[i]))
                for i, r in enumerate(test_rows)}
    write_predictions(args.output, test_rows, fallback)
    try:
        features = embed_sentences(
            train_text + test_text, resolve_model_path(args.model_dir),
            args.batch_size, args.max_length, args.device,
            deadline=start + args.budget_seconds - 45.0,
        )
    except BudgetExceeded as error:
        print(f"{error}; kept legal-boundary fallback predictions", file=sys.stderr)
        return
    from sklearn.linear_model import LogisticRegression
    train_x, test_x = features[:len(train_text)], features[len(train_text):]
    fit_mask = np.isin(train_article, train_idx) & (y >= 0)
    if np.unique(y[fit_mask]).size != 2:
        raise ValueError("Training needs both human and machine sentence labels")
    classifier = LogisticRegression(C=args.regularization_c, max_iter=150,
                                    solver="lbfgs", random_state=args.seed)
    classifier.fit(train_x[fit_mask], y[fit_mask])
    if validation_idx:
        probabilities = classifier.predict_proba(train_x)[:, 1]
        decoded = decode_articles(train_rows, train_spans, train_article,
                                  probabilities, mean_fraction, args)
        val_answers = {train_rows[i]["id"]: answers[train_rows[i]["id"]] for i in validation_idx}
        print(f"Article-holdout score: {boundary_score(decoded, val_answers):.4f} "
              "(local split, NOT official leaderboard)", file=sys.stderr)
    if time.monotonic() < start + args.budget_seconds - 20:
        classifier.fit(train_x[y >= 0], y[y >= 0])
        mean_fraction = float(np.mean([answers[r["id"]] / max(1, len(r["text"])) for r in train_rows]))
    probabilities = classifier.predict_proba(test_x)[:, 1]
    predictions = decode_articles(test_rows, test_spans, test_article,
                                  probabilities, mean_fraction, args)
    write_predictions(args.output, test_rows, predictions)
    print(f"Wrote {len(predictions)} predictions in {time.monotonic()-start:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
