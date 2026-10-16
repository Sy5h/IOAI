"""Select a simple Potato policy with PUBLIC-only synthetic judges.

No private embeddings or released A/B targets are read. This is development,
not an official score. The candidate list and selection metric are fixed here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
import numpy as np

CANDIDATES = {
    "baseline": {},
    "guess_only": {"guess_after": 1},
    "guess_after_8": {"guess_after": 8},
    "guess_after_12": {"guess_after": 12},
    "direct_bonus_3": {"bonus": 3.0},
    "forget_095": {"retention": 0.95},
    "forget_090": {"retention": 0.90},
    "floor_020": {"floor": 0.20},
    "mixture": {"temperatures": [0.03, 0.06, 0.12], "floor": 0.12},
    "mixture_guess_8": {"temperatures": [0.03, 0.06, 0.12], "floor": 0.12, "guess_after": 8},
    "vote": {"vote": True},
    "vote_guess_8": {"vote": True, "guess_after": 8},
}


def load_module(path):
    spec = importlib.util.spec_from_file_location("round2_potato_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def player_class(base, config):
    class Player(base.PublicEmbeddingPlayer):
        def __init__(self, words, similarities):
            super().__init__(words, similarities, error_floor=config.get("floor", 0.08))
            self.temperatures = config.get("temperatures", [0.06])
            self.components = np.zeros((len(self.temperatures), len(words)), dtype=np.float64)

        def observe(self, message):
            first = self.word_to_index[message["word1"].casefold()]
            second = self.word_to_index[message["word2"].casefold()]
            self.components *= config.get("retention", 1.0)
            for k, temperature in enumerate(self.temperatures):
                if config.get("vote") and message["verdict"] != "same":
                    sign = 1 if message["verdict"] == "first" else -1
                    margin = sign * (self.similarities[:, first] - self.similarities[:, second])
                    evidence = np.sign(margin) * 0.75
                else:
                    evidence = base.comparison_log_likelihood(
                        self.similarities, first, second, message["verdict"],
                        temperature, self.error_floor)
                self.components[k] += evidence
            self.log_weight[:] = self.components[0]

        def posterior(self):
            values = self.components.copy()
            if self.proposed:
                values[:, list(self.proposed)] = -np.inf
            values = np.exp(values - values.max(axis=1, keepdims=True))
            values /= values.sum(axis=1, keepdims=True)
            return values.mean(axis=0)

        def choose(self, champion, turn):
            posterior = self.posterior()
            ranked = np.argsort(-posterior)
            best = int(ranked[0])
            if turn >= config.get("guess_after", 24) or posterior[best] >= (0.18 if turn >= 16 else 0.35):
                return best
            support = ranked[:min(self.support_size, len(ranked))]
            support = support[posterior[support] > 0]
            mass = posterior[support] / posterior[support].sum()
            close = np.argsort(-self.similarities[champion])[:32]
            spread = np.linspace(0, len(self.words) - 1, min(64, len(self.words))).astype(int)
            queries = np.unique(np.concatenate([ranked[:64], close, spread]))
            queries = queries[posterior[queries] > 0]
            margin = self.similarities[np.ix_(support, queries)] - self.similarities[support, champion, None]
            probability = self.error_floor + (1 - 2 * self.error_floor) * base.sigmoid(margin / self.temperature)
            information = base.binary_entropy(mass @ probability) - mass @ base.binary_entropy(probability)
            hit = config.get("bonus", 1.0) * (0.5 + turn / 12.0) * posterior[queries]
            return int(queries[np.argmax(information + hit)])
    return Player


def simulate(player, words, public_similarity, judge_similarity, targets):
    index = {word: i for i, word in enumerate(words)}
    scores, turns = [], []
    started = time.monotonic()
    for secret in targets:
        instance = player(words, public_similarity)
        first, second = index["lamp"], index["potato"]
        found = None
        for turn in range(1, 31):
            difference = float(judge_similarity[secret, first] - judge_similarity[secret, second])
            verdict = "same" if abs(difference) <= 1e-12 else ("first" if difference > 0 else "second")
            champion = second if verdict == "second" else first
            proposal = index[instance.respond({"turn": turn, "word1": words[first], "word2": words[second],
                                               "winner_word": words[champion], "verdict": verdict})]
            if proposal == secret:
                found = turn
                break
            first, second = champion, proposal
        turns.append(found)
        scores.append(0.0 if found is None else 1.0 - 0.02 * max(0, found - 10))
    return {"score": 100 * float(np.mean(scores)), "wins": sum(t is not None for t in turns),
            "games": len(targets), "simulation_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--exclude-development", help="Prior development JSON: exclude its sampled words")
    parser.add_argument("--candidates", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    args = parser.parse_args()
    base = load_module(args.baseline)
    data = Path(args.data_dir)
    words, similarity = base.load_resources(data)
    public_targets = set(json.loads((data / "test_public.json").read_text(encoding="utf-8")))
    pool = [i for i, w in enumerate(words) if w not in public_targets]
    if args.exclude_development:
        previous = json.loads(Path(args.exclude_development).read_text(encoding="utf-8"))
        excluded = set(np.random.default_rng(previous["seed"]).choice(
            pool, min(previous["games_per_judge"], len(pool)), replace=False).tolist())
        pool = [i for i in pool if i not in excluded]
    rng = np.random.default_rng(args.seed)
    targets = rng.choice(pool, min(args.count, len(pool)), replace=False)
    embedding = np.load(data / "public_embeddings.npy").astype(np.float32)
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    judges = {"public_exact": similarity}
    for dimensions in (128, 32):
        projection = rng.normal(size=(embedding.shape[1], dimensions)).astype(np.float32)
        projected = embedding @ projection
        projected /= np.linalg.norm(projected, axis=1, keepdims=True)
        judges[f"public_projection_{dimensions}"] = projected @ projected.T
    projected = embedding @ rng.normal(size=(embedding.shape[1], 128)).astype(np.float32)
    projected += 0.5 * rng.normal(size=projected.shape).astype(np.float32)
    projected /= np.linalg.norm(projected, axis=1, keepdims=True)
    judges["public_projection_128_noise_05"] = projected @ projected.T
    result = {"label": "PUBLIC-only synthetic development, NOT an official score",
              "baseline_commit": "991a49b48092286967b5bd7cada465e336ebc4af",
              "data_revision": "122716fc56a212e58b1756158bdeb4c2ea2da5e6",
              "seed": args.seed, "games_per_judge": len(targets),
              "excluded_prior_development": bool(args.exclude_development),
              "target_indices_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
              "selection_rule": "Mean of the three perturbed judges; exact public score is descriptive only.",
              "candidates": {}, "threads": 4}
    for name in args.candidates:
        candidate = player_class(base, CANDIDATES[name])
        record = {"parameters": CANDIDATES[name], "judges": {}}
        for judge, matrix in judges.items():
            record["judges"][judge] = simulate(candidate, words, similarity, matrix, targets)
        record["robust_mean"] = float(np.mean([r["score"] for k, r in record["judges"].items() if k != "public_exact"]))
        result["candidates"][name] = record
        print(json.dumps({"candidate": name, **record}), flush=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["selected"] = max(result["candidates"], key=lambda n: result["candidates"][n]["robust_mean"])
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SELECTED", result["selected"], flush=True)


if __name__ == "__main__":
    main()
