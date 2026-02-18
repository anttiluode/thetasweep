"""
Example: Document Retrieval
============================
Demonstrates SweepRetriever on a sample email corpus.
Zero pretrained models. Zero external ML dependencies.
"""

import time
from thetasweep import SweepRetriever

# Sample email corpus
sample_emails = """
From: Random AI
Subject: Party this weekend
Date: 2024-01-15

Hi Antti! Don't forget to bring the Turkish Peppers and energy drinks.

From: Antti Luode
Subject: PerceptionLab eigenmode results
Date: 2024-02-01

The Phi-Dwell analysis is complete. AD brains show MORE eigenmode vocabulary
(1052 vs 953 words) with LESS structure - lower Zipf slope, reduced top-5
concentration. The dwell gradient delta-to-gamma is the strongest biomarker
(p=0.0015, rho=0.408 with MMSE score). Alpha-band stability is the key differentiator.

From: Scientist Number One
Subject: Re: PerceptionLab eigenmode results
Date: 2024-02-02

The Alzheimer's connection is striking. If the theta clock degrades,
the sweep becomes diffuse, positions blur - exactly what your reservoir
experiments showed when the gate sigma was too large.

From: Antti Luode
Subject: MoireFormer - Reverse task breakthrough
Date: 2024-02-10

Sweep attractor architecture: 100% on Copy, Reverse, Shift.
The key: sequence is a TRAJECTORY through phase space, not a static vector.
Forward sweep = left theta sweep. Backward sweep = right theta sweep.
Random projection beats Gabor/Grid geometry (98% vs 83%).
Training takes 200ms via ridge regression. Transformer needs 13 seconds.
"""

print("ThetaSweep: Document Retrieval Demo")
print("=" * 45)
print("Zero pretrained models. Privacy-preserving.")
print()

retriever = SweepRetriever(stack_size=512, sigma=1.5)
retriever.build_index(sample_emails)

print()
queries = [
    "Don't forget",
    "what are the Alzheimer's findings",
    "Eigenlab results",
    "sweep architecture results",
    "The dwell gradient",
]

for q in queries:
    t0 = time.perf_counter()
    hits = retriever.retrieve(q, top_k=1)
    ms = (time.perf_counter() - t0) * 1000
    chunk, score = hits[0]
    preview = chunk[:120].replace('\n', ' ')
    print(f"Q: {q}")
    print(f"A: [{score:.3f}] {preview}...")
    print(f"   ({ms:.1f}ms)")
    print()
