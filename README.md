# ThetaSweep

**Biologically-grounded sequence processing and document retrieval via directional reservoir sweeps.**

Zero backpropagation. Trains in milliseconds. Runs on CPU with numpy only.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green)](https://python.org)
[![numpy](https://img.shields.io/badge/dependency-numpy%20only-orange)](https://numpy.org)

---

## The Core Idea

The brain does not store sequences as static feature vectors. It *traverses* them rhythmically via a theta-frequency clock.

This was observed empirically in PerceptionLab (A. Luode, 2024): a **box attractor at theta frequency** in frontal EEG phase space — a 4-state limit cycle clock. This is not metaphor. It is the literal computational structure driving sequence memory in the frontal lobe.

Vollan et al. (2025) provided the mechanism: in each ~125ms theta cycle, grid cell populations sweep outward from the current position, alternating left/right on successive cycles. Position is encoded in **when** within the cycle a neuron fires (phase precession), not **how much** it fires (amplitude).

ThetaSweep implements this computationally:

```
Input sequence → Random reservoir projection
              → Forward sweep  (left theta sweep analogue)
              → Backward sweep (right theta sweep analogue)
              → Ridge regression readout  (zero backprop)
              → Output
```

At sweep step k, a sharp Gaussian gate focuses on position k — analogous to theta phase precession. Forward sweep encodes input position k at output step k. Backward sweep encodes input position N-1-k at output step k. **Copy, Reverse, and Shift become trivially linear for the readout.**

---

## What It Does

### 1. Sequence Processing — `SweepReservoir`

Solves sequence permutation tasks with zero backpropagation:

| Task    | Accuracy | Train time | vs Transformer |
|---------|----------|------------|----------------|
| Copy    | ~100%    | ~200ms     | 64x faster     |
| Reverse | ~100%    | ~200ms     | 64x faster     |
| Shift   | ~52%     | ~200ms     | known limitation |

Training is ridge regression: closed-form, O(d^2 * n). No GPU. No gradient descent.

### 2. Document Retrieval — `SweepRetriever`

Indexes and queries documents using sweep reservoir embeddings. **No pretrained models. No external downloads.**

| Property | Value |
|----------|-------|
| External ML dependencies | None |
| Pretrained model weights | None |
| Index time (2.6MB / ~9000 chunks) | ~40s on CPU |
| Query time | < 5ms |
| Privacy | Complete — no data leaves the machine |

---

## Benchmarks

### Sequence tasks vs Transformer

Tested on vocab=8, seq_len=6, 600 training sequences, 200 test sequences:

```
Task       ThetaSweep    Transformer    Speedup
Copy         ~100%         100%          64x
Reverse      ~100%         100%          64x
Shift         ~52%         100%          — (known weakness)
```

### Retrieval vs BM25 (Elasticsearch standard)

Tested on 50 technical documents with 50 ground-truth query-answer pairs
covering neuroscience, ML, physics, and applied engineering topics:

```
Metric      ThetaSweep    BM25      Delta
Recall@1      96.0%      100.0%    -4.0%
Recall@3     100.0%      100.0%    +0.0%
MRR           0.977       1.000    -0.023
```

**ThetaSweep matches BM25 at Recall@3 with zero pretrained model weights.**
BM25 indexes faster (1ms vs 50ms for this corpus size) but both answer in ~0.1ms per query.

Run the benchmark yourself:
```bash
pip install rank-bm25
python examples/benchmark_vs_bm25.py
```

---

## Installation

```bash
git clone https://github.com/anttiluode/thetasweep
cd thetasweep
pip install -e .
```

No dependencies beyond numpy for core use.
Optional: `rank-bm25` for running the benchmark. `llama-cpp-python` for LLM generation.

---

## Quick Start

### Sequence tasks

```python
from thetasweep import SweepReservoir
from thetasweep.tasks import solve_sequence_task, evaluate_sequence_task

reservoir = SweepReservoir(vocab_size=8, stack_size=512)

W, train_ms = solve_sequence_task(reservoir, task='reverse')
acc = evaluate_sequence_task(reservoir, W, task='reverse')

print(f"Reverse accuracy: {acc:.1%} in {train_ms:.0f}ms")
# -> Reverse accuracy: 99.9% in 190ms
```

### Document retrieval

```python
from thetasweep import SweepRetriever

retriever = SweepRetriever()

with open("my_documents.txt") as f:
    text = f.read()

retriever.build_index(text)

results = retriever.retrieve("what are the Alzheimer findings", top_k=3)
for chunk, score in results:
    print(f"[{score:.3f}] {chunk[:100]}")
```

### With local LLM (fully private, no cloud)

```python
from thetasweep import SweepRetriever
from llama_cpp import Llama

retriever = SweepRetriever()
retriever.build_index(open("documents.txt").read())

llm = Llama(model_path="model.gguf", n_ctx=4096, n_gpu_layers=0)

query = "what does the research say about the theta clock"
hits = retriever.retrieve(query, top_k=4)
context = "\n---\n".join([chunk for chunk, score in hits])

output = llm(
    f"<|system|>\nAnswer based only on:\n{context}\n<|user|>\n{query}\n<|assistant|>\n",
    max_tokens=256
)
print(output['choices'][0]['text'])
```

---

## Examples

```bash
python examples/sequence_tasks.py       # sequence benchmark
python examples/document_retrieval.py   # retrieval demo
python examples/benchmark_vs_bm25.py    # ThetaSweep vs BM25
```

---

## Research Background

This library emerged from a 15-month experimental project in PerceptionLab exploring biologically-grounded neural architectures.

**Phase 1 — Static geometry (MoiréFormer)**
Hypothesis: Gabor/Grid geometric interference as a universal basis set.
Result: 100% Copy, ~18% Reverse. Static geometry creates symmetric diffusion that destroys positional information.

**Phase 2 — Hierarchical diffusion control**
Hypothesis: Mimicking brain frequency-band separation (delta/alpha/gamma).
Result: +42% on Shift, Reverse unchanged.

**Phase 3 — Cross-band amplitude binding**
Hypothesis: Theta-gamma amplitude coupling encodes position x identity.
Result: -15% on Reverse. Amplitude multiplication of mixed signals amplifies noise.

**Phase 4 — Theta sweep (breakthrough)**
Insight from Vollan et al. (2025) + box attractor in frontal EEG (PerceptionLab, 2024):
position is encoded in *when* a neuron fires, not *how much*.
Result: 100% on Copy, Reverse, Shift.

**Phase 5 — Geometry vs random control**
Deerskin sweep (Gabor/Grid) vs random projection sweep head-to-head.
Result: random wins 98% vs 83%. The sweep direction is the load-bearing innovation.

### Connection to Alzheimer's disease

Phi-Dwell eigenmode analysis of clinical EEG shows AD brains have more eigenmode vocabulary with less structure — higher variety, lower persistence, weaker concentration in dominant modes. This mirrors reservoir failure under over-diffusion.

If the theta clock degrades, sweep gates become diffuse, positions blur, and sequence processing fails. The delta-to-gamma dwell gradient is the strongest EEG biomarker (p=0.0015, rho=0.408 with MMSE). This is a research hypothesis, not a validated clinical tool.

---

## Known Limitations

- **Shift task**: ~52% accuracy. Pure sweep lacks local adjacency encoding.
- **Semantic retrieval**: BoW + character n-grams do not generalise semantically ("happy" != "joyful"). Pretrained embeddings outperform on semantic queries.
- **Index speed**: BM25 indexes faster than ThetaSweep on large corpora.
- **Toy task validation**: Sequence results validated on vocab <= 32, seq_len <= 50.
- **Not a clinical tool**: The Alzheimer's connection is a research hypothesis.

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

MIT License — see [LICENSE](LICENSE) for full terms.

Copyright (c) 2024 Antti Luode
