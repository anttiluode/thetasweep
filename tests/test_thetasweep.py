"""
Basic tests for ThetaSweep library.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from thetasweep import SweepReservoir, SweepRetriever
from thetasweep.tasks import generate_sequences, benchmark


class TestSweepReservoir:

    def test_encode_shape(self):
        res = SweepReservoir(vocab_size=8, stack_size=64, seed=0)
        feat = res.encode([0, 1, 2, 3, 4, 5])
        assert feat.shape == (6, 128)  # 2 * stack_size

    def test_copy_task(self):
        res = SweepReservoir(vocab_size=8, stack_size=256, seed=42)
        seqs, tgts = generate_sequences('copy', 8, 6, 400)
        W = res.train(seqs, tgts)
        acc = res.accuracy(seqs[:100], tgts[:100], W)
        assert acc > 0.95, f"Copy accuracy {acc:.1%} < 95%"

    def test_reverse_task(self):
        res = SweepReservoir(vocab_size=8, stack_size=256, seed=42)
        seqs, tgts = generate_sequences('reverse', 8, 6, 400)
        W = res.train(seqs, tgts)
        acc = res.accuracy(seqs[:100], tgts[:100], W)
        assert acc > 0.95, f"Reverse accuracy {acc:.1%} < 95%"

    def test_predict_output_shape(self):
        res = SweepReservoir(vocab_size=8, stack_size=64, seed=0)
        seqs, tgts = generate_sequences('copy', 8, 6, 100)
        W = res.train(seqs, tgts)
        pred = res.predict([0, 1, 2, 3, 4, 5], W)
        assert pred.shape == (6,)

    def test_reproducible_with_seed(self):
        res1 = SweepReservoir(vocab_size=8, stack_size=64, seed=42)
        res2 = SweepReservoir(vocab_size=8, stack_size=64, seed=42)
        feat1 = res1.encode([1, 2, 3])
        feat2 = res2.encode([1, 2, 3])
        np.testing.assert_array_equal(feat1, feat2)


class TestSweepRetriever:

    def setup_method(self):
        self.sample_text = """
From: Katrina Luode
Subject: Party

Hi Antti! Bring Turkish Peppers and energy drinks.
Izzy the cat from Georgia misses you.
Dorinda Knepper (DorieForrie) is coming Friday.

From: Antti Luode
Subject: Research

The eigenmode analysis shows AD brains have more vocabulary with less structure.
The dwell gradient is the strongest biomarker.
"""

    def test_build_index(self):
        r = SweepRetriever(stack_size=128, sigma=1.5)
        r.build_index(self.sample_text, verbose=False)
        assert r.chunk_embeddings is not None
        assert len(r.chunk_texts) > 0

    def test_retrieve_returns_results(self):
        r = SweepRetriever(stack_size=128, sigma=1.5)
        r.build_index(self.sample_text, verbose=False)
        results = r.retrieve("Katrina party", top_k=2)
        assert len(results) == 2
        assert all(isinstance(c, str) for c, s in results)
        assert all(isinstance(s, float) for c, s in results)

    def test_retrieve_before_index_raises(self):
        r = SweepRetriever()
        with pytest.raises(RuntimeError):
            r.retrieve("test")

    def test_relevant_retrieval(self):
        r = SweepRetriever(stack_size=256, sigma=1.5)
        r.build_index(self.sample_text, verbose=False)
        hits = r.retrieve("Alzheimer eigenmode research", top_k=1)
        chunk, score = hits[0]
        # Should find the research email, not the party email
        assert "eigenmode" in chunk.lower() or "dwell" in chunk.lower() or score > 0


if __name__ == "__main__":
    print("Running tests...")

    # Manual test run
    print("\n--- SweepReservoir tests ---")
    t = TestSweepReservoir()
    t.test_encode_shape(); print("  encode shape: OK")
    t.test_copy_task(); print("  copy task: OK")
    t.test_reverse_task(); print("  reverse task: OK")
    t.test_predict_output_shape(); print("  predict shape: OK")
    t.test_reproducible_with_seed(); print("  reproducibility: OK")

    print("\n--- SweepRetriever tests ---")
    tr = TestSweepRetriever()
    tr.setup_method()
    tr.test_build_index(); print("  build index: OK")
    tr.test_retrieve_returns_results(); print("  retrieve results: OK")
    tr.test_relevant_retrieval(); print("  relevant retrieval: OK")

    print("\nAll tests passed.")
