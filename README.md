# ThetaSweep

**A hybrid between a search engine and a biological brain simulation.**

ThetaSweep processes information the way the hippocampus does: not as a static list of items, but as a trajectory through time. It is a sweep-based reservoir computer inspired by theta phase precession in the entorhinal-hippocampal circuit.

Zero backpropagation. Trains in milliseconds. Runs on CPU with numpy only.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green)](https://python.org)
[![numpy](https://img.shields.io/badge/dependency-numpy%20only-orange)](https://numpy.org)

---

## What It Is

ThetaSweep is a **Sweep-Based Reservoir Computer** with three components:

### 1. The Reservoir (The Frozen Brain)
A high-dimensional echo chamber of thousands of random neurons. Unlike GPT or Llama, the connections are **fixed and random — nothing is learned here**. Input data is projected into this space where different patterns land in different regions. The randomness is deliberate: high-dimensional random spaces are excellent at keeping things separable.

### 2. The Sweep (The Clock)
This is the core invention. Instead of processing all input at once, the system **scans rhythmically** — like a radar beam, or like the theta oscillations in your hippocampus:

- **Forward Sweep**: scans input positions left → right
- **Backward Sweep**: scans input positions right → left

At each sweep step k, a sharp Gaussian gate focuses on position k and attenuates everything else. This is the computational analogue of **theta phase precession**: position is encoded in *when* the gate focuses, not *how strongly* a neuron fires.

This was not invented from theory. It was discovered empirically: a **box attractor at theta frequency** was observed in frontal EEG phase space using PerceptionLab (A. Luode, 2024) — a 4-state limit cycle clock. Vollan et al. (2025) then provided the biological mechanism: alternating left/right theta sweeps in the entorhinal-hippocampal circuit encoding prospective and retrospective positions.

### 3. The Readout (The Answer)
A single matrix of weights trained via **ridge regression** — closed-form linear algebra, solved in one shot. No gradient descent. No training epochs. Because the sweep has already organized the data in time, the readout problem is trivially linear.

---

## What It Does

### A. Solves Sequence Tasks Instantly

Standard AI needs thousands of training examples and minutes of backpropagation to learn how to reverse a list. ThetaSweep solves it in milliseconds with zero training epochs.

The backward sweep pre-solves the reversal by construction: at sweep step k, the backward gate focuses on position N-1-k. The readout just reads the backward sweep states in order — no inversion needed, it's already there.

| Task    | Accuracy | Train time | vs Transformer |
|---------|----------|------------|----------------|
| Copy    | ~100%    | ~200ms     | 64x faster     |
| Reverse | ~100%    | ~200ms     | 64x faster     |
| Shift   | ~52%     | ~200ms     | known limitation |

### B. Acts as a Semantic Compass for Documents

Feed it a text file — emails, a book, research notes, anything — and it converts every chunk into coordinates in the reservoir's high-dimensional space. Then search it.

```
[Theta-Nav] >> i wonder about dorinda

Timeline: [▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒·▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒·▒▒▒▒▒▒▒▒▒▒·▒▒▒▒·▒▒▒▒▒·] (965 hits in 4.1ms)

Top Hit (0.235):
From: dorieforrie@bellsouth.net
Sent: Friday, August 30, 2024 2:52 PM
To: Antti Luode <anttinorthwest@hotmail.com>...
```

Searching 2.6MB of personal email (9652 chunks), finding the right person in 4ms, entirely locally. No cloud. No Google. No data leaving the machine.

**Context-aware**: because of the sweep gate, each chunk carries a memory of its neighbors in document order. Word X is stored with an echo of words X-1 and X+1.

**Timeline visualization**: `readbigfile.py` maps where hits appear across the full document — a spatial view of where a concept lives in your archive, or your life history.

### C. Benchmarked Against the Industry Standard

Tested against BM25 — the retrieval algorithm used inside Elasticsearch — on 50 technical documents with ground-truth query-answer pairs:

```
Metric      ThetaSweep    BM25 (Elasticsearch)
Recall@1      96.0%          100.0%
Recall@3     100.0%          100.0%
MRR           0.977           1.000
Query time    0.1ms           0.1ms
```

**Matches BM25 at Recall@3 with zero pretrained model weights.**

---

## How It Differs from Standard AI

| Property | Transformers / LLMs | ThetaSweep |
|----------|---------------------|------------|
| Parameters | Billions, all learned | ~0 learned (readout only) |
| Training | Weeks on GPUs | Milliseconds on CPU |
| Memory required | 4GB+ | Hundreds of MB |
| Semantic understanding | Yes (learned from internet) | No |
| Privacy | Data sent to cloud | Fully local |
| Interpretable | No (black box) | Yes (inspect every weight) |
| Generalisation | Broad | Narrow but exact |
| Dependency | CUDA, huge models | numpy only |

Standard AI compresses statistical patterns from vast datasets into billions of weights. It "understands" meaning because it has seen most of human writing.

ThetaSweep constructs a geometric structure where the answer is already easy to read, then uses the simplest possible mathematics to extract it. It does not understand meaning. It understands *position in time and space*.

They are not competing tools. They solve different problems.

---

## What It Might Be Used For

**Personal archive search** — emails, notes, documents, chat logs. Instant local search over years of personal data with no cloud dependency. Export from Gmail, feed to ThetaSweep, query privately forever.

**Edge intelligence** — sensors, embedded systems, Raspberry Pi. Index a pattern library in milliseconds. Detect anomalies against that library in sub-millisecond query time. No model download. No internet required.

**Research tools** — index a corpus of papers, navigate by concept. The timeline visualization shows where ideas cluster across a document collection.

**RAG without the cloud** — combine with a local GGUF model (Llama, Mistral, Phi) for fully private question-answering over personal documents. ThetaSweep handles retrieval, the local LLM handles generation.

**Signal processing** — the sweep reservoir can process time-series signals, not just text. Anomaly detection, pattern matching, classification on sensor data with millisecond training.

---

## Connection to Alzheimer's Disease Research

The same mechanism that breaks ThetaSweep also appears to break human memory in Alzheimer's disease.

Phi-Dwell eigenmode analysis of clinical EEG (Luode, 2024) shows AD brains have **more eigenmode vocabulary with less structure** — higher variety of brain states, lower persistence in any one state, weaker dominance of the top modes. This mirrors exactly what happens to a reservoir computer under over-diffusion: expanded vocabulary, reduced persistence, loss of structured dynamics.

If the theta clock degrades — as it does in early Alzheimer's — the sweep gates become diffuse, positions blur together, and sequence processing fails. The delta-to-gamma dwell gradient is the strongest EEG biomarker found (p=0.0015, ρ=0.408 with MMSE cognitive score).

This is a research hypothesis connecting the computational model to the clinical finding. It is not a validated diagnostic tool.

---

## Installation

```bash
git clone https://github.com/anttiluode/thetasweep
cd thetasweep
pip install -e .
```

No dependencies beyond numpy for core use.

---

## Quick Start

```bash
# Sequence tasks
python examples/sequence_tasks.py

# Document retrieval demo
python examples/document_retrieval.py

# Navigate a large file interactively
python examples/readbigfile.py your_document.txt

# Benchmark vs BM25 (requires: pip install rank-bm25)
python examples/benchmark_vs_bm25.py
```

### In your own code

```python
from thetasweep import SweepRetriever

retriever = SweepRetriever()
retriever.build_index(open("my_emails.txt").read())

results = retriever.retrieve("what did katrina say about the party", top_k=3)
for chunk, score in results:
    print(f"[{score:.3f}] {chunk[:100]}")
```

```python
from thetasweep import SweepReservoir
from thetasweep.tasks import solve_sequence_task, evaluate_sequence_task

reservoir = SweepReservoir(vocab_size=8, stack_size=512)
W, ms = solve_sequence_task(reservoir, task='reverse')
acc = evaluate_sequence_task(reservoir, W, task='reverse')
print(f"Reverse: {acc:.1%} in {ms:.0f}ms")
```

---

## Known Limitations

- **Shift task**: ~52% accuracy. The sweep lacks local adjacency encoding.
- **Semantic understanding**: does not generalise ("happy" ≠ "joyful"). Pretrained embeddings outperform on meaning-based queries.
- **Index speed**: BM25 indexes faster on large corpora (1ms vs 40s for ~10k chunks).
- **Scale**: ridge regression is O(d² × n). Very large stack sizes on very large corpora become expensive.
- **Not a clinical tool**: the Alzheimer's connection is a research hypothesis.

---

## References

- Vollan, H.R. et al. (2025). Theta sweeps in the entorhinal-hippocampal circuit encode prospective and retrospective positions. *Nature*.
- Luode, A. (2024). Phi-Dwell: Eigenmode dwell time analysis of EEG differentiates Alzheimer's disease from healthy controls. PerceptionLab.
- Luode, A. (2024). Box attractor at theta frequency in frontal EEG phase space. PerceptionLab observation log.

---

## Citation

```bibtex
@software{luode2024thetasweep,
  author  = {Luode, Antti},
  title   = {ThetaSweep: Biologically-grounded sequence processing via directional reservoir sweeps},
  year    = {2024},
  url     = {https://github.com/anttiluode/thetasweep},
  note    = {Inspired by theta phase precession in entorhinal-hippocampal circuits (Vollan et al., 2025)}
}
```

---

## License

MIT — see [LICENSE](LICENSE).  
Copyright (c) 2024 Antti Luode
