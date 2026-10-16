"""Potato: a soft-posterior search player using only the supplied embeddings.

Self-contained: this file can be the code cell of solution.ipynb.  The official
grader converts that notebook to Python.  Importing/running a notebook cell does
not enter the stdin loop; executing this .py file does.  Diagnostics use stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def binary_entropy(p):
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return -p * np.log(p) - (1.0 - p) * np.log1p(-p)


def comparison_log_likelihood(similarities, first, second, verdict,
                              temperature=0.06, error_floor=0.08):
    """One likelihood per possible secret; positive margin favors `first`.

    A near-zero margin carries little evidence.  The floor acknowledges that
    public and private embedding spaces disagree; nothing is hard-eliminated.
    """
    if temperature <= 0 or not 0 < error_floor < 0.5:
        raise ValueError("temperature > 0 and 0 < error_floor < 0.5 required")
    margin = similarities[:, first] - similarities[:, second]
    if verdict == "first":
        agreement = sigmoid(margin / temperature)
    elif verdict == "second":
        agreement = sigmoid(-margin / temperature)
    elif verdict == "same":
        if first == second:
            return np.zeros(similarities.shape[0], dtype=np.float64)
        agreement = np.exp(-np.abs(margin) / temperature)
    else:
        raise ValueError(f"Unknown verdict: {verdict!r}")
    return np.log(error_floor + (1.0 - 2.0 * error_floor) * agreement)


def load_resources(data_dir):
    data_dir = Path(data_dir)
    with (data_dir / "vocabulary.json").open(encoding="utf-8") as f:
        words = json.load(f)
    embeddings = np.load(data_dir / "public_embeddings.npy").astype(np.float32)
    if embeddings.ndim != 2 or len(words) != embeddings.shape[0]:
        raise ValueError("Vocabulary and embedding rows do not match")
    if len({w.casefold() for w in words}) != len(words):
        raise ValueError("Vocabulary contains duplicate words")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(embeddings).all() or np.any(norms <= 0):
        raise ValueError("Embeddings must be finite, nonzero vectors")
    embeddings /= norms
    similarities = embeddings @ embeddings.T
    return words, similarities


_SHARED = None


class PublicEmbeddingPlayer:
    """A fresh instance per game; all instances share the similarity matrix."""

    def __init__(self, words=None, similarities=None, temperature=0.06,
                 error_floor=0.12, support_size=256, deadline=None,
                 posterior_mode="mixture", guess_after=8):
        global _SHARED
        if words is None or similarities is None:
            if _SHARED is None:
                _SHARED = load_resources(os.environ.get("POTATO_DATA_DIR", "dataset"))
            words, similarities = _SHARED
        self.words = list(words)
        self.similarities = np.asarray(similarities, dtype=np.float32)
        self.word_to_index = {w.casefold(): i for i, w in enumerate(self.words)}
        if self.similarities.shape != (len(words), len(words)):
            raise ValueError("Expected a square vocabulary similarity matrix")
        if posterior_mode not in {"single", "mixture"}:
            raise ValueError("posterior_mode must be single or mixture")
        if not np.isfinite(temperature) or temperature <= 0 or not 0 < error_floor < 0.5:
            raise ValueError("finite temperature > 0 and 0 < error_floor < 0.5 required")
        if not 1 <= guess_after <= 30:
            raise ValueError("guess_after must be between 1 and 30")
        self.temperatures = ([temperature] if posterior_mode == "single"
                             else [temperature / 2, temperature, temperature * 2])
        self.component_log_weight = np.zeros((len(self.temperatures), len(words)), dtype=np.float64)
        self.log_weight = self.component_log_weight[0]  # single-mode compatibility
        self.proposed = set()
        self.temperature = temperature
        self.error_floor = error_floor
        self.support_size = max(1, support_size)
        self.deadline = deadline
        self.guess_after = guess_after

    def posterior(self):
        weights = self.component_log_weight.copy()
        if self.proposed:
            weights[:, list(self.proposed)] = -np.inf
        if not np.isfinite(weights).any():
            raise RuntimeError("No unused vocabulary words remain")
        weights = np.exp(weights - weights.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        # Mix normalized posteriors: a confident component cannot suppress the
        # more cautious components merely because its likelihood scale differs.
        return weights.mean(axis=0)

    def observe(self, message):
        first = self.word_to_index[message["word1"].casefold()]
        second = self.word_to_index[message["word2"].casefold()]
        for component, temperature in zip(self.component_log_weight, self.temperatures):
            component += comparison_log_likelihood(
                self.similarities, first, second, message["verdict"],
                temperature, self.error_floor,
            )

    def choose(self, champion, turn):
        posterior = self.posterior()
        ranked = np.argsort(-posterior)
        best = int(ranked[0])
        # Fixed using public-only synthetic development before released A/B.
        if (turn >= self.guess_after or posterior[best] >= (0.18 if turn >= 16 else 0.35)
                or (self.deadline is not None and time.monotonic() > self.deadline - 15)):
            return best

        support = ranked[:min(self.support_size, len(ranked))]
        support = support[posterior[support] > 0]
        mass = posterior[support]
        mass = mass / mass.sum()
        # A bounded query set keeps 120 games comfortably smaller than running
        # a transformer or evaluating every word against every hypothesis each turn.
        close = np.argsort(-self.similarities[champion])[:32]
        spread = np.linspace(0, len(self.words) - 1, min(64, len(self.words))).astype(int)
        queries = np.unique(np.concatenate([ranked[:64], close, spread]))
        queries = queries[posterior[queries] > 0]
        margin = (self.similarities[np.ix_(support, queries)]
                  - self.similarities[support, champion, None])
        response_prob = (self.error_floor + (1 - 2 * self.error_floor)
                         * sigmoid(margin / self.temperature))
        # Mutual information discounts queries that are a coin flip for EVERY
        # hypothesis, unlike simply trying to balance the mean response at 1/2.
        mean_response = mass @ response_prob
        information = binary_entropy(mean_response) - mass @ binary_entropy(response_prob)
        direct_hit_bonus = (0.5 + turn / 12.0) * posterior[queries]
        return int(queries[np.argmax(information + direct_hit_bonus)])

    def respond(self, message):
        self.observe(message)
        champion = self.word_to_index[message["winner_word"].casefold()]
        proposal = self.choose(champion, int(message["turn"]))
        # If respond() is called again, this guess necessarily failed. Initial
        # lamp/potato are NOT put here: seeing them is not an explicit guess.
        self.proposed.add(proposal)
        return self.words[proposal]


def run_interactive(player_factory=PublicEmbeddingPlayer, input_stream=None,
                    output_stream=None):
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    player = None
    for line in input_stream:
        if not line.strip():
            continue
        message = json.loads(line)
        if message.get("event") == "done":
            break
        if message.get("event") == "new_game":
            player = player_factory()
            continue
        if "status" in message:
            continue
        if player is None:
            player = player_factory()
        word = player.respond(message)
        print(json.dumps({"new_word": word}), file=output_stream, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("POTATO_DATA_DIR", "dataset"))
    parser.add_argument("--temperature", type=float, default=0.06)
    parser.add_argument("--error-floor", type=float, default=0.12)
    parser.add_argument("--posterior-mode", choices=["single", "mixture"], default="mixture")
    parser.add_argument("--guess-after", type=int, default=8)
    parser.add_argument("--support-size", type=int, default=256)
    parser.add_argument("--budget-seconds", type=float, default=580.0)
    args = parser.parse_args()
    start = time.monotonic()
    global _SHARED
    _SHARED = load_resources(args.data_dir)
    print(f"Prepared {len(_SHARED[0])} words in {time.monotonic()-start:.2f}s", file=sys.stderr)
    run_interactive(lambda: PublicEmbeddingPlayer(
        *_SHARED, temperature=args.temperature, error_floor=args.error_floor,
        support_size=args.support_size, deadline=start + args.budget_seconds,
        posterior_mode=args.posterior_mode, guess_after=args.guess_after,
    ))


if __name__ == "__main__" and "__file__" in globals():
    main()
