"""Synthetic contract/algorithm tests; not an official contest benchmark."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "solutions" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


potato = load("potato_test_module", "day1_potato.py")
ghost = load("ghost_test_module", "day2_ghost_of_the_machine.py")


class PotatoTests(unittest.TestCase):
    def setUp(self):
        self.words = ["lamp", "potato", "rock", "shovel"]
        embeddings = np.array([[1, 0], [0, 1], [0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
        self.similarity = embeddings @ embeddings.T

    def test_comparison_direction_and_strength(self):
        first = potato.comparison_log_likelihood(self.similarity, 0, 1, "first")
        second = potato.comparison_log_likelihood(self.similarity, 0, 1, "second")
        self.assertGreater(first[0], first[1])
        self.assertGreater(second[1], second[0])
        self.assertGreater(first[0], first[2])
        self.assertTrue(np.isfinite(first).all())  # a disagreement is never impossible

    def test_same_is_neutral_for_repeated_word_and_favors_similar_distances(self):
        neutral = potato.comparison_log_likelihood(self.similarity, 0, 0, "same")
        self.assertTrue(np.all(neutral == 0))
        same = potato.comparison_log_likelihood(self.similarity, 0, 1, "same")
        self.assertGreater(same[2], same[0])

    def test_initial_words_remain_candidates_and_failed_guesses_are_removed(self):
        player = potato.PublicEmbeddingPlayer(self.words, self.similarity)
        message = {"turn": 1, "word1": "lamp", "word2": "potato",
                   "winner_word": "lamp", "verdict": "first"}
        player.observe(message)
        self.assertGreater(player.posterior()[0], 0)
        self.assertGreater(player.posterior()[1], 0)
        guess = player.respond(message)
        self.assertEqual(player.posterior()[self.words.index(guess)], 0)

    def test_protocol_resets_player_and_only_emits_json_guesses(self):
        messages = [
            {"event": "new_game"},
            {"turn": 1, "word1": "lamp", "word2": "potato", "winner_word": "lamp", "verdict": "first"},
            {"status": "win"},
            {"event": "new_game"},
            {"turn": 1, "word1": "lamp", "word2": "potato", "winner_word": "lamp", "verdict": "first"},
            {"status": "win"}, {"event": "done"},
        ]
        created = []

        def factory():
            player = potato.PublicEmbeddingPlayer(self.words, self.similarity)
            created.append(player)
            return player

        output = io.StringIO()
        potato.run_interactive(factory, io.StringIO("\n".join(map(json.dumps, messages))), output)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(created), 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0], records[1])
        self.assertEqual(set(records[0]), {"new_word"})

    def test_complete_small_games_with_public_space_as_simulated_judge(self):
        # This verifies the interactive state progression, not private-space quality.
        for secret in range(len(self.words)):
            player = potato.PublicEmbeddingPlayer(self.words, self.similarity)
            first, second = 0, 1
            guesses = []
            for turn in range(1, len(self.words) + 1):
                delta = self.similarity[secret, first] - self.similarity[secret, second]
                verdict = "same" if abs(delta) < 1e-12 else ("first" if delta > 0 else "second")
                champion = second if verdict == "second" else first
                word = player.respond({"turn": turn, "word1": self.words[first],
                                       "word2": self.words[second], "winner_word": self.words[champion],
                                       "verdict": verdict})
                proposal = self.words.index(word)
                self.assertNotIn(proposal, guesses)
                guesses.append(proposal)
                if proposal == secret:
                    break
                first, second = champion, proposal
            self.assertIn(secret, guesses)


class GhostTests(unittest.TestCase):
    def test_no_internal_punctuation_decodes_to_legal_mean_word_boundary(self):
        text = " ".join(["word"] * 600) + "."
        spans = ghost.sentence_spans(text, [". "])
        self.assertEqual(spans, [(0, len(text))])
        for probabilities in [[0.01], [0.99]]:
            for expected_score in [False, True]:
                prediction = ghost.choose_boundary(
                    text, spans, probabilities, mean_fraction=0.6,
                    expected_score=expected_score,
                )
                self.assertEqual(prediction, 1800)
                self.assertEqual(len(text[:prediction].split()), 360)
                self.assertEqual(len(text[prediction:].split()), 240)
        self.assertEqual(ghost.fallback_boundary(text, 0.0), 900)
        self.assertEqual(ghost.fallback_boundary(text, 1.0), 2100)

    def test_budget_fallback_for_no_punctuation_preserves_legal_character_offset(self):
        text = " ".join(["word"] * 600) + "."
        with tempfile.TemporaryDirectory() as directory:
            train_dir, test_dir = Path(directory) / "train", Path(directory) / "test"
            train_dir.mkdir()
            test_dir.mkdir()
            (train_dir / "data.jsonl").write_text(
                json.dumps({"id": "train", "text": text}), encoding="utf-8")
            (train_dir / "answers.jsonl").write_text(
                json.dumps({"id": "train", "boundary_char_index": 1800}), encoding="utf-8")
            (test_dir / "data.jsonl").write_text(
                json.dumps({"id": "test", "text": text}), encoding="utf-8")
            output = Path(directory) / "answers.jsonl"
            with patch.object(ghost, "embed_sentences", side_effect=ghost.BudgetExceeded("test deadline")):
                ghost.main(["--train-dir", str(train_dir), "--test-dir", str(test_dir),
                            "--output", str(output)])
            prediction = ghost.read_jsonl(output)[0]["boundary_char_index"]
            self.assertEqual(prediction, 1800)
            self.assertGreaterEqual(len(text[:prediction].split()), 180)
            self.assertGreaterEqual(len(text[prediction:].split()), 180)
            self.assertEqual(text[prediction:prediction+4], "word")

    def test_end_to_end_pipeline_with_controlled_sentence_embeddings(self):
        try:
            import sklearn  # noqa: F401 -- the real classifier is exercised
        except ImportError:
            self.skipTest("scikit-learn not installed")
        with tempfile.TemporaryDirectory() as directory:
            train_dir, test_dir = Path(directory) / "train", Path(directory) / "test"
            train_dir.mkdir()
            test_dir.mkdir()
            rows = [{"id": str(i), "text": f"Human a{i}. Human b{i}. Machine c{i}. Machine d{i}."}
                    for i in range(8)]
            answers = [{"id": r["id"], "boundary_char_index": r["text"].index("Machine")} for r in rows]
            (train_dir / "data.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            (train_dir / "answers.jsonl").write_text("\n".join(map(json.dumps, answers)), encoding="utf-8")
            test_rows = [{"id": "test", "text": "Human first. Human second. Machine third. Machine last."}]
            (test_dir / "data.jsonl").write_text(json.dumps(test_rows[0]), encoding="utf-8")
            output = Path(directory) / "answers.jsonl"

            def controlled_embeddings(sentences, *args, **kwargs):
                return np.asarray([[1.0, 0.0] if s.lstrip().startswith("Human") else [0.0, 1.0]
                                   for s in sentences])

            with patch.object(ghost, "embed_sentences", side_effect=controlled_embeddings):
                ghost.main(["--train-dir", str(train_dir), "--test-dir", str(test_dir),
                            "--output", str(output), "--min-side-words", "0", "--prior-strength", "0"])
            self.assertEqual(ghost.read_jsonl(output), [{"id": "test", "boundary_char_index": 27}])

    def test_separator_learning_respects_boundary_before_or_after_space(self):
        texts = ["A human sentence. Machine begins here. More words.",
                 "Another human sentence. Machine begins here too. More."]
        rows = [{"id": str(i), "text": text} for i, text in enumerate(texts)]
        before = {str(i): text.index(". ") + 1 for i, text in enumerate(texts)}
        after = {str(i): text.index(". ") + 2 for i, text in enumerate(texts)}
        self.assertEqual(ghost.learn_separators(rows, before), ["."])
        self.assertEqual(ghost.learn_separators(rows, after), [". "])
        for answers in [before, after]:
            separators = ghost.learn_separators(rows, answers)
            for row in rows:
                spans = ghost.sentence_spans(row["text"], separators)
                self.assertIn(answers[row["id"]], [start for start, _ in spans])
                self.assertEqual("".join(row["text"][start:end] for start, end in spans), row["text"])

    def test_single_switch_ignores_an_isolated_suspicious_sentence(self):
        probabilities = [0.02, 0.8, 0.02, 0.02, 0.98, 0.98]
        self.assertEqual(int(np.argmax(ghost.one_switch_logits(probabilities))), 4)

    def test_unicode_offsets_and_minimum_word_constraint(self):
        text = "사람 글이다. 사람 다음이다. 기계 글이다. 기계 다음이다."
        spans = ghost.sentence_spans(text, [". "])
        prediction = ghost.choose_boundary(text, spans, [0.01, 0.01, 0.99, 0.99],
                                           prior_strength=0, min_side_words=0)
        self.assertEqual(prediction, text.index("기계"))
        self.assertEqual(text[prediction:][:2], "기계")
        short_side = ghost.choose_boundary(text, spans, [0.99] * 4,
                                           prior_strength=0, min_side_words=3)
        self.assertGreaterEqual(len(text[:short_side].split()), 3)
        self.assertGreaterEqual(len(text[short_side:].split()), 3)

    def test_article_validation_has_no_shared_articles(self):
        rows = [{"id": str(i), "text": "text"} for i in range(20)]
        train, validation = ghost.article_split(rows)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(range(20)))

    def test_crossing_sentence_is_not_taught_a_wrong_label(self):
        rows = [{"id": "a", "text": "Human half then machine half."}]
        _, labels, _, _ = ghost.build_sentence_data(rows, [". "], {"a": 11})
        self.assertEqual(labels.tolist(), [-1])

    def test_output_contract_and_metric(self):
        rows = [{"id": "a", "text": "abc def"}, {"id": "b", "text": "xyz"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            ghost.write_predictions(path, rows, {"a": 4, "b": 0})
            self.assertEqual(ghost.read_jsonl(path)[0], {"id": "a", "boundary_char_index": 4})
            with self.assertRaises(ValueError):
                ghost.write_predictions(path, rows, {"a": 4})
            with self.assertRaises(ValueError):
                ghost.write_predictions(path, rows, {"a": 8, "b": 0})
            with self.assertRaises(ValueError):
                ghost.write_predictions(path, rows, {"a": 4.5, "b": 0})
        self.assertAlmostEqual(ghost.boundary_score({"a": 100}, {"a": 0}), 100 * math_exp_minus_one())


def math_exp_minus_one():
    return float(np.exp(-1))


if __name__ == "__main__":
    unittest.main()
