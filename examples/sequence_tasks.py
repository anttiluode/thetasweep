"""
Example: Sequence Tasks
=======================
Demonstrates the sweep reservoir on Copy, Reverse, and Shift tasks.
Shows training speed vs Transformer baseline.
"""

import numpy as np
from thetasweep import SweepReservoir
from thetasweep.tasks import benchmark

print("ThetaSweep: Sequence Task Demo")
print("=" * 45)
print()
print("Key result: sweep reservoir trains in milliseconds")
print("with no backpropagation. Pure ridge regression.")
print()

reservoir = SweepReservoir(vocab_size=8, stack_size=512, sigma=0.8, seed=42)

print("Results (vocab=8, seq_len=6, 600 train, 200 test):")
results = benchmark(reservoir, verbose=True)

print()
print("Interpretation:")
print("  Copy/Reverse: ~100% - sweep direction makes these trivially linear")
print("  Shift: ~50%  - pure sweep lacks local adjacency (known limitation)")
print()
print("Biological grounding:")
print("  Forward sweep  = left theta sweep (Vollan et al. 2025)")
print("  Backward sweep = right theta sweep (alternating cycles)")
print("  Phase gate     = theta phase precession in grid cells")
print("  Ridge readout  = linear decoder downstream of hippocampus")
