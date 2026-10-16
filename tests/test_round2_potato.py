"""Algorithm contracts for the public-only round-two Potato policy."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "round2_potato", Path(__file__).resolve().parents[1] / "solutions/day1_potato.py")
potato = importlib.util.module_from_spec(spec)
spec.loader.exec_module(potato)


class RobustPotatoTests(unittest.TestCase):
    def setUp(self):
        self.words = ["lamp", "potato", "rock", "shovel"]
        embedding = np.array([[1, 0], [0, 1], [.8, .6], [.6, .8]], dtype=np.float32)
        self.similarity = embedding @ embedding.T

    def test_cautious_component_keeps_mass_after_another_component_is_confident(self):
        player = potato.PublicEmbeddingPlayer(self.words, self.similarity)
        player.component_log_weight[0] = [0, -1000, -1000, -1000]
        player.component_log_weight[1] = [0, 0, 0, 0]
        player.component_log_weight[2] = [-1000, 0, -1000, -1000]
        posterior = player.posterior()
        self.assertAlmostEqual(posterior.sum(), 1.)
        self.assertGreater(posterior[1], .4)
        self.assertGreater(posterior[2], .08)
        self.assertTrue(np.isfinite(posterior).all())

    def test_failed_guess_is_removed_from_every_component_but_initial_words_remain(self):
        player = potato.PublicEmbeddingPlayer(self.words, self.similarity)
        player.proposed.add(2)
        player.component_log_weight[:, 2] = 1000
        posterior = player.posterior()
        self.assertEqual(posterior[2], 0.)
        self.assertGreater(posterior[0], 0.)
        self.assertGreater(posterior[1], 0.)

    def test_eighth_turn_directly_guesses_the_best_unused_candidate(self):
        player = potato.PublicEmbeddingPlayer(self.words, self.similarity)
        player.component_log_weight[:] = [0, 0, .1, .2]
        player.proposed.add(3)
        self.assertEqual(player.choose(0, 8), 2)
        self.assertEqual(player.choose(0, 30), 2)

    def test_single_mode_reproduces_accumulated_likelihood_without_mixture(self):
        player = potato.PublicEmbeddingPlayer(self.words, self.similarity,
                                               posterior_mode="single", error_floor=.08, guess_after=24)
        message = {"word1": "lamp", "word2": "potato", "verdict": "first"}
        player.observe(message)
        expected = np.exp(potato.comparison_log_likelihood(self.similarity, 0, 1, "first"))
        np.testing.assert_allclose(player.posterior(), expected / expected.sum(), rtol=1e-6)

    def test_invalid_configuration_fails_instead_of_emitting_invalid_probabilities(self):
        for options in ({"temperature": float("nan")}, {"posterior_mode": "unknown"},
                        {"guess_after": 0}, {"guess_after": 31}, {"error_floor": .5}):
            with self.assertRaises(ValueError):
                potato.PublicEmbeddingPlayer(self.words, self.similarity, **options)


if __name__ == "__main__":
    unittest.main()
