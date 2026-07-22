"""Tests for evaluate_pipeline.py — Sprint 2-3 metric functions."""

import json
import os
import sys
import unittest

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate_pipeline as ep


class TestReduceGolden(unittest.TestCase):
    """Sprint 3a: Golden sample reduction."""

    def test_keeps_n_samples(self):
        rows = [{"question": str(i), "ground_truth": str(i)} for i in range(15)]
        reduced = ep.reduce_golden(rows, n=5)
        self.assertEqual(len(reduced), 5)

    def test_samples_are_diverse(self):
        rows = [{"question": str(i), "ground_truth": str(i)} for i in range(15)]
        reduced = ep.reduce_golden(rows, n=5)
        questions = [r["question"] for r in reduced]
        # Should be 5 unique questions
        self.assertEqual(len(set(questions)), 5)

    def test_handles_smaller_input(self):
        rows = [{"question": str(i)} for i in range(3)]
        reduced = ep.reduce_golden(rows, n=5)
        self.assertLessEqual(len(reduced), 3)

    def test_handles_empty(self):
        self.assertEqual(len(ep.reduce_golden([], n=5)), 0)


class TestLimitContexts(unittest.TestCase):
    """Sprint 3b: Context limiting for 2B model."""

    def test_caps_at_k(self):
        rows = [{"contexts": list(range(10))}]
        limited = ep.limit_contexts(rows, max_ctx=3)
        self.assertEqual(len(limited[0]["contexts"]), 3)

    def test_preserves_first_k(self):
        rows = [{"contexts": ["a", "b", "c", "d", "e"]}]
        limited = ep.limit_contexts(rows, max_ctx=2)
        self.assertEqual(limited[0]["contexts"], ["a", "b"])

    def test_no_op_when_under_k(self):
        rows = [{"contexts": ["a", "b"]}]
        limited = ep.limit_contexts(rows, max_ctx=5)
        self.assertEqual(limited[0]["contexts"], ["a", "b"])

    def test_handles_missing_contexts(self):
        rows = [{"question": "test"}]
        limited = ep.limit_contexts(rows, max_ctx=3)
        self.assertEqual(limited[0].get("contexts", []), [])


class TestContextRecall(unittest.TestCase):
    """Sprint 2: Context recall with word overlap."""

    def test_perfect_match(self):
        gt = "SYNTH_MAX_TOKENS_OUT=384"
        contexts = ["SYNTH_MAX_TOKENS_OUT=384 was set in config.py"]
        score = ep.context_recall_local(gt, contexts)
        self.assertGreater(score, 0.5)

    def test_no_match(self):
        gt = "this ground truth has enough tokens"
        contexts = ["completely unrelated text here"]
        score = ep.context_recall_local(gt, contexts)
        self.assertLess(score, 0.1)

    def test_partial_match(self):
        gt = "384 was 250 before v1.0.7"
        contexts = ["SYNTH_MAX_TOKENS_OUT=384 (was 250 before v1.0.7)"]
        score = ep.context_recall_local(gt, contexts)
        self.assertGreater(score, 0.5)

    def test_empty_ground_truth(self):
        self.assertEqual(ep.context_recall_local("", ["some context"]), 0.0)

    def test_empty_contexts(self):
        self.assertEqual(ep.context_recall_local("some truth", []), 0.0)

    def test_short_gt_falls_back_to_cosine(self):
        """Very short ground truth (< 4 tokens) uses cosine similarity."""
        gt = "faithfulness"
        contexts = ["faithfulness score is 1.0"]
        score = ep.context_recall_local(gt, contexts)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestContextPrecisionSentence(unittest.TestCase):
    """Sprint 2: Sentence-level context precision."""

    def test_empty_contexts(self):
        self.assertEqual(
            ep.context_precision_local_sentence("q", "gt", []),
            0.0
        )

    def test_returns_float(self):
        ctx = ["This is a test sentence. It has two sentences."]
        score = ep.context_precision_local_sentence("test", "test", ctx)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestAnswerRelevancyDecompose(unittest.TestCase):
    """Sprint 2: Decomposition-based answer relevancy."""

    def test_empty_answer(self):
        self.assertEqual(
            ep.answer_relevancy_local_decompose("test", ""),
            0.0
        )

    def test_empty_question(self):
        self.assertEqual(
            ep.answer_relevancy_local_decompose("", "test"),
            0.0
        )

    def test_returns_float(self):
        score = ep.answer_relevancy_local_decompose(
            "what does the API.md document describe?",
            "The API.md document describes the REST endpoints."
        )
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_template_question_detects_doc_name(self):
        score = ep.answer_relevancy_local_decompose(
            "what does the BENCHMARKS.md document describe?",
            "The BENCHMARKS document describes extraction benchmarks."
        )
        self.assertGreater(score, 0.2)  # At least doc_name check passes


if __name__ == "__main__":
    unittest.main()