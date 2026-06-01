import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.metrics.diversity import distinct_n, self_bleu, tokenize
from mop_divpo.metrics.semantic import (
    cross_group_mean_cosine,
    mean_pairwise_cosine,
    pairwise_cosine_matrix,
)
from mop_divpo.divpo.pairs import select_pair


class DiversityTests(unittest.TestCase):
    def test_distinct_n_all_unique_is_one(self):
        texts = ["alpha beta", "gamma delta"]
        self.assertEqual(distinct_n(texts, 1), 1.0)

    def test_distinct_1_counts_repeats(self):
        # 4 tokens total, 1 unique ("cat") -> 0.25
        self.assertEqual(distinct_n(["cat cat", "cat cat"], 1), 0.25)
        # 4 tokens total, 2 unique ("cat","dog") -> 0.5
        self.assertEqual(distinct_n(["cat dog", "cat dog"], 1), 0.5)

    def test_distinct_n_returns_zero_when_texts_shorter_than_n(self):
        self.assertEqual(distinct_n(["word"], 2), 0.0)

    def test_self_bleu_identical_texts_is_high(self):
        identical = ["the cat sat on the mat", "the cat sat on the mat"]
        diverse = ["the cat sat on the mat", "quantum physics explains particle behavior"]
        self.assertGreater(self_bleu(identical), self_bleu(diverse))

    def test_self_bleu_single_text_is_zero(self):
        self.assertEqual(self_bleu(["only one"]), 0.0)

    def test_self_bleu_identical_is_near_one(self):
        score = self_bleu(["the cat sat on the mat today", "the cat sat on the mat today"])
        self.assertGreater(score, 0.9)

    def test_tokenize_lowercases_and_strips_punctuation(self):
        self.assertEqual(tokenize("Hello, World!"), ["hello", "world"])


class SemanticTests(unittest.TestCase):
    def test_pairwise_matrix_diagonal_is_one(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        matrix = pairwise_cosine_matrix(embs)
        self.assertAlmostEqual(matrix[0, 0], 1.0, places=5)
        self.assertAlmostEqual(matrix[1, 1], 1.0, places=5)
        self.assertAlmostEqual(matrix[0, 1], 0.0, places=5)

    def test_mean_pairwise_cosine_orthogonal_is_zero(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.assertAlmostEqual(mean_pairwise_cosine(embs), 0.0, places=5)

    def test_mean_pairwise_cosine_identical_is_one(self):
        embs = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        self.assertAlmostEqual(mean_pairwise_cosine(embs), 1.0, places=5)

    def test_mean_pairwise_cosine_single_vector_is_zero(self):
        self.assertEqual(mean_pairwise_cosine(np.array([[1.0, 0.0]], dtype=np.float32)), 0.0)

    def test_cross_group_mean_cosine_orthogonal_groups(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        self.assertAlmostEqual(cross_group_mean_cosine(a, b), 0.0, places=5)

    def test_cross_group_mean_cosine_empty_group_is_zero(self):
        a = np.zeros((0, 2), dtype=np.float32)
        b = np.array([[1.0, 0.0]], dtype=np.float32)
        self.assertEqual(cross_group_mean_cosine(a, b), 0.0)


class DivPOPairTests(unittest.TestCase):
    def test_select_pair_rejects_pairs_below_rarity_margin(self):
        class FakeEmbedder:
            pass

        pair = select_pair(
            "prompt",
            ["candidate alpha words", "candidate beta words"],
            FakeEmbedder(),
            min_quality=0.0,
            min_rarity_margin=0.5,
            _prompt_emb=np.array([1.0, 0.0], dtype=np.float32),
            _cand_embs=[
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.9, 0.1], dtype=np.float32),
            ],
        )

        self.assertIsNone(pair)


if __name__ == "__main__":
    unittest.main()
