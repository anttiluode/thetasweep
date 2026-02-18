"""
ThetaSweep vs BM25: Honest Retrieval Benchmark
================================================
BM25 is the standard baseline for information retrieval.
It's what Elasticsearch uses. Most RAG pipelines use it.

We test both on a corpus of 50 technical documents with
ground-truth query-answer pairs. We measure:
  - Recall@1: was the right chunk the top result?
  - Recall@3: was the right chunk in the top 3?
  - MRR: mean reciprocal rank (standard IR metric)

The corpus covers: neuroscience, AI/ML, and general science topics
so it's not trivially easy keyword matching.
"""

import numpy as np
import re
import sys
import time
sys.path.insert(0, '.')

from thetasweep import SweepRetriever
from rank_bm25 import BM25Okapi


# ==============================================================================
# TEST CORPUS - 50 documents with ground truth query→doc mappings
# ==============================================================================

CORPUS = [
    # Neuroscience / EEG / Brain
    "Theta oscillations in the hippocampus occur at 4-8 Hz and are strongly linked to spatial navigation and episodic memory encoding. Grid cells in the entorhinal cortex fire at specific phases of the theta cycle.",
    "Phase precession is a phenomenon where place cells fire at progressively earlier phases of the theta cycle as an animal traverses the cell's place field. This creates a temporal code for position.",
    "The box attractor is a limit cycle in phase space with four corners, representing a discrete clock cycling through four states. Observed in frontal EEG at theta frequency during planning tasks.",
    "Alzheimer's disease is associated with tau protein tangles and amyloid plaques. Early symptoms include short-term memory loss and difficulty with spatial navigation.",
    "Eigenmode decomposition of EEG reveals the dominant spatial patterns of brain activity. Dwell time in each eigenmode reflects the stability of that brain state.",
    "The default mode network is active during rest and mind-wandering. It is deactivated during focused cognitive tasks and includes the medial prefrontal cortex and posterior cingulate.",
    "Alpha oscillations at 8-12 Hz are associated with inhibition of task-irrelevant cortical regions. Increased alpha power correlates with reduced neural excitability.",
    "Gamma oscillations above 30 Hz are linked to local cortical computation and feature binding. Theta-gamma coupling coordinates information across hippocampal-prefrontal circuits.",
    "The entorhinal cortex is the main interface between the hippocampus and neocortex. It receives highly processed sensory information and projects to hippocampal CA1 via the perforant path.",
    "Neuronal avalanches are cascades of activity that follow a power law distribution, suggesting the brain operates near criticality for optimal information transmission.",

    # Reservoir Computing / ML
    "Echo state networks are a form of reservoir computing where a fixed random recurrent network projects inputs into a high-dimensional space. Only the readout layer is trained.",
    "Ridge regression minimizes the sum of squared errors plus a penalty on the L2 norm of the weights. It has a closed-form solution: W = (X'X + λI)^{-1} X'Y.",
    "Backpropagation through time (BPTT) computes gradients for recurrent networks by unrolling the network across time steps. It suffers from vanishing and exploding gradients.",
    "The transformer architecture uses self-attention to compute relationships between all positions in a sequence simultaneously. Training requires large datasets and significant compute.",
    "Random projections preserve pairwise distances approximately according to the Johnson-Lindenstrauss lemma. A random matrix with entries drawn from N(0,1/d) is sufficient.",
    "Liquid state machines are a reservoir computing model where a recurrent spiking network serves as the reservoir. They were proposed by Maass et al. as a model of cortical computation.",
    "The kernel trick allows linear algorithms to operate in high-dimensional feature spaces without explicitly computing the transformation. The RBF kernel is equivalent to an infinite-dimensional feature space.",
    "Catastrophic forgetting occurs when a neural network trained on new tasks loses performance on previously learned tasks. Elastic weight consolidation attempts to mitigate this.",
    "BM25 is a bag-of-words retrieval function based on term frequency and inverse document frequency. It is the standard baseline for information retrieval and used in Elasticsearch.",
    "Sentence transformers produce dense vector embeddings of text using contrastive learning. Models like all-MiniLM-L6-v2 are trained on millions of sentence pairs.",

    # Physics / Math
    "A Gaussian function is defined as f(x) = exp(-x^2 / 2σ^2). It achieves maximum value at x=0 and decays smoothly to zero. The standard deviation σ controls the width.",
    "Fourier transforms decompose a signal into its constituent frequencies. The FFT algorithm computes this in O(n log n) time, enabling fast spectral analysis.",
    "The Lorenz attractor is a strange attractor in a three-dimensional dynamical system that exhibits chaotic behavior. Small differences in initial conditions lead to divergent trajectories.",
    "Eigenvalues of a matrix A satisfy det(A - λI) = 0. The eigenvectors define directions that are only scaled, not rotated, by the linear transformation.",
    "Interference patterns arise when waves from multiple sources superimpose. Constructive interference occurs when waves are in phase; destructive interference when they are out of phase.",
    "The central limit theorem states that the sum of independent random variables approaches a normal distribution as n increases, regardless of the underlying distribution.",
    "Gradient descent updates parameters in the direction of the negative gradient of the loss function. The learning rate controls the step size and affects convergence stability.",
    "Convolution of two functions f and g is defined as (f*g)(t) = integral of f(τ)g(t-τ)dτ. In CNNs, learned filters are convolved with input feature maps.",
    "Phase space is the space of all possible states of a dynamical system. A trajectory in phase space represents the evolution of the system over time.",
    "Zipf's law states that the frequency of the nth most common item is proportional to 1/n. It applies to word frequencies in natural language, city populations, and neural firing rates.",

    # Biology / Neuroscience continued
    "NMDA receptors are voltage-gated ion channels that require both ligand binding and membrane depolarization to open. They are critical for synaptic plasticity and LTP.",
    "Long-term potentiation (LTP) is a persistent strengthening of synaptic connections following high-frequency stimulation. It is considered a cellular mechanism of learning and memory.",
    "Dendritic computation allows individual neurons to perform complex nonlinear operations. Dendritic spines can act as independent computational subunits.",
    "The basal ganglia are involved in motor control, procedural learning, and reward-based decision making. They implement a form of reinforcement learning through dopaminergic signaling.",
    "Myelin sheaths insulate axons and dramatically increase conduction velocity. Demyelinating diseases like multiple sclerosis disrupt the timing of neural signals.",
    "The cerebellum contains approximately 69 billion neurons, more than the rest of the brain combined. It is involved in fine motor control, timing, and predictive coding.",
    "Action potentials are all-or-nothing electrical signals propagated along axons. The refractory period prevents backward propagation and sets a maximum firing rate.",
    "Astrocytes regulate synaptic transmission by controlling extracellular glutamate levels. They also contribute to the blood-brain barrier and provide metabolic support to neurons.",
    "The hippocampus is critical for the formation of new episodic memories. Bilateral hippocampal lesions, as in patient H.M., result in severe anterograde amnesia.",
    "Parkinson's disease results from loss of dopaminergic neurons in the substantia nigra. It is characterized by tremor, rigidity, and bradykinesia.",

    # Applied / Engineering
    "TF-IDF (term frequency-inverse document frequency) weights terms by how often they appear in a document relative to how often they appear across all documents.",
    "Cosine similarity measures the angle between two vectors regardless of magnitude. It is widely used for comparing text embeddings in information retrieval.",
    "The bag-of-words model represents text as an unordered collection of word counts. It ignores grammar and word order but is effective for topic classification.",
    "Named entity recognition (NER) identifies and classifies entities in text such as person names, organizations, and locations. It is a core NLP task.",
    "Retrieval-augmented generation (RAG) combines a retrieval system with a language model. Relevant documents are retrieved and included in the prompt to ground the model's responses.",
    "Vector databases store high-dimensional embeddings and support approximate nearest-neighbor search. Examples include Pinecone, Weaviate, and FAISS.",
    "Chunking strategies for document retrieval include fixed-size windows, sentence boundaries, and paragraph splits. Overlap between chunks preserves context at boundaries.",
    "Quantization reduces model size by representing weights with lower precision. INT8 quantization can reduce model size by 4x with minimal accuracy loss.",
    "Edge computing processes data near the source rather than in centralized servers. This reduces latency and bandwidth requirements for IoT and sensor applications.",
    "Zero-shot learning enables models to perform tasks without task-specific training examples by leveraging semantic descriptions or analogies to known classes.",
]

# Ground truth: (query, correct_doc_index)
GROUND_TRUTH = [
    ("theta oscillations hippocampus memory", 0),
    ("phase precession place cells theta", 1),
    ("box attractor limit cycle theta frequency", 2),
    ("Alzheimer's disease tau amyloid memory", 3),
    ("eigenmode decomposition EEG brain states", 4),
    ("default mode network rest deactivation", 5),
    ("alpha oscillations inhibition brain", 6),
    ("gamma oscillations theta coupling hippocampus", 7),
    ("entorhinal cortex hippocampus interface", 8),
    ("neuronal avalanches power law criticality", 9),
    ("echo state network reservoir computing readout", 10),
    ("ridge regression closed form solution", 11),
    ("backpropagation through time vanishing gradient", 12),
    ("transformer self-attention sequence", 13),
    ("random projection Johnson-Lindenstrauss", 14),
    ("liquid state machines spiking reservoir cortical", 15),
    ("kernel trick RBF infinite dimensional", 16),
    ("catastrophic forgetting elastic weight consolidation", 17),
    ("BM25 term frequency inverse document frequency retrieval", 18),
    ("sentence transformers dense embeddings contrastive", 19),
    ("Gaussian function standard deviation width", 20),
    ("Fourier transform FFT spectral analysis", 21),
    ("Lorenz attractor chaotic dynamical system", 22),
    ("eigenvalues eigenvectors linear transformation", 23),
    ("interference waves constructive destructive", 24),
    ("central limit theorem normal distribution", 25),
    ("gradient descent learning rate convergence", 26),
    ("convolution CNN feature maps filters", 27),
    ("phase space trajectory dynamical system", 28),
    ("Zipf law frequency distribution language", 29),
    ("NMDA receptors synaptic plasticity LTP", 30),
    ("long-term potentiation synaptic strengthening memory", 31),
    ("dendritic computation nonlinear subunits", 32),
    ("basal ganglia reinforcement learning dopamine", 33),
    ("myelin sheath conduction velocity demyelination", 34),
    ("cerebellum fine motor control timing", 35),
    ("action potential refractory period firing rate", 36),
    ("astrocytes glutamate blood brain barrier", 37),
    ("hippocampus episodic memory anterograde amnesia", 38),
    ("Parkinson's disease dopamine substantia nigra", 39),
    ("TF-IDF term frequency document weighting", 40),
    ("cosine similarity vector embeddings retrieval", 41),
    ("bag of words word counts topic classification", 42),
    ("named entity recognition NER person organization", 43),
    ("retrieval augmented generation RAG language model", 44),
    ("vector database nearest neighbor FAISS Pinecone", 45),
    ("chunking document retrieval overlap context", 46),
    ("quantization INT8 model compression", 47),
    ("edge computing IoT latency sensor", 48),
    ("zero-shot learning semantic analogies", 49),
]


# ==============================================================================
# BM25 WRAPPER
# ==============================================================================

class BM25Retriever:
    def __init__(self):
        self.corpus = None
        self.bm25 = None

    def build_index(self, chunks):
        self.corpus = chunks
        tokenized = [re.findall(r'\b[a-zA-Z]+\b', c.lower()) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, top_k=5):
        tokens = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        scores = self.bm25.get_scores(tokens)
        top_idx = scores.argsort()[-top_k:][::-1]
        return [(self.corpus[i], float(scores[i])) for i in top_idx]


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate(retriever, ground_truth, corpus, top_k=3, model_name="Model"):
    recall_1 = recall_3 = mrr = 0
    n = len(ground_truth)

    t0 = time.perf_counter()
    for query, correct_idx in ground_truth:
        hits = retriever.retrieve(query, top_k=top_k)
        retrieved_chunks = [h[0] for h in hits]
        correct_chunk = corpus[correct_idx]

        # Recall@1
        if retrieved_chunks[0] == correct_chunk:
            recall_1 += 1

        # Recall@3 and MRR
        for rank, chunk in enumerate(retrieved_chunks, 1):
            if chunk == correct_chunk:
                recall_3 += 1
                mrr += 1.0 / rank
                break

    t_total = (time.perf_counter() - t0) * 1000

    r1 = recall_1 / n
    r3 = recall_3 / n
    mrr_score = mrr / n

    print(f"\n  {model_name}")
    print(f"  {'─'*40}")
    print(f"  Recall@1:  {r1:.1%}  ({recall_1}/{n} correct as top result)")
    print(f"  Recall@3:  {r3:.1%}  ({recall_3}/{n} correct in top 3)")
    print(f"  MRR:       {mrr_score:.3f}")
    print(f"  Query time:{t_total:.1f}ms total ({t_total/n:.1f}ms avg)")

    return r1, r3, mrr_score


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 55)
    print("  THETASWEEP vs BM25: RETRIEVAL BENCHMARK")
    print("  50 documents, 50 ground-truth query-answer pairs")
    print("  Standard IR metrics: Recall@1, Recall@3, MRR")
    print("=" * 55)

    # Build both indexes
    print("\nBuilding indexes...")

    t0 = time.perf_counter()
    sweep = SweepRetriever(stack_size=512, sigma=1.5)
    sweep.build_index(CORPUS, verbose=False)
    t_sweep = (time.perf_counter() - t0) * 1000
    print(f"  ThetaSweep index: {t_sweep:.0f}ms")

    t0 = time.perf_counter()
    bm25 = BM25Retriever()
    bm25.build_index(CORPUS)
    t_bm25 = (time.perf_counter() - t0) * 1000
    print(f"  BM25 index:       {t_bm25:.0f}ms")

    # Evaluate
    print("\nResults:")
    r1_s, r3_s, mrr_s = evaluate(sweep, GROUND_TRUTH, CORPUS, model_name="ThetaSweep")
    r1_b, r3_b, mrr_b = evaluate(bm25, GROUND_TRUTH, CORPUS, model_name="BM25 (Elasticsearch standard)")

    # Summary
    print(f"\n{'=' * 55}")
    print(f"  SUMMARY")
    print(f"{'=' * 55}")
    print(f"  {'Metric':<12} {'ThetaSweep':>12} {'BM25':>10} {'Delta':>10}")
    print(f"  {'─'*44}")
    print(f"  {'Recall@1':<12} {r1_s:>11.1%} {r1_b:>9.1%} {r1_s-r1_b:>+9.1%}")
    print(f"  {'Recall@3':<12} {r3_s:>11.1%} {r3_b:>9.1%} {r3_s-r3_b:>+9.1%}")
    print(f"  {'MRR':<12} {mrr_s:>11.3f} {mrr_b:>9.3f} {mrr_s-mrr_b:>+9.3f}")

    print(f"\n  Index time: ThetaSweep {t_sweep:.0f}ms vs BM25 {t_bm25:.0f}ms")

    print(f"\n  INTERPRETATION:")
    if r1_s >= r1_b - 0.05:
        print(f"  ✓ ThetaSweep matches BM25 on factual retrieval.")
        print(f"    For a zero-pretrained-model approach, this is strong.")
    elif r1_s >= r1_b - 0.15:
        print(f"  ~ ThetaSweep within 15% of BM25.")
        print(f"    Competitive for a non-standard approach.")
    else:
        print(f"  ✗ BM25 leads significantly.")
        print(f"    ThetaSweep retriever needs improvement.")

    print(f"\n  NOTE: BM25 is purely lexical (same word = match).")
    print(f"  ThetaSweep adds character n-grams and sweep context.")
    print(f"  Neither does semantic matching (happy ≠ joyful).")
    print(f"  Both are zero-pretrained-model approaches.")


if __name__ == "__main__":
    main()
