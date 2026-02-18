"""
SweepRetriever: Document retrieval via sweep reservoir.

Zero pretrained models. Zero external ML dependencies beyond numpy.
Indexes documents using bag-of-words + character n-gram features,
projected through a random reservoir with directional sweeps.

The sweep gate provides context-awareness: each chunk's embedding
is influenced by its neighboring chunks in document order, so
"what Katrina said in paragraph 3" carries context from paragraphs
2 and 4, not just its own words.

Compared to pretrained embedding models (MiniLM, etc.):
    Slower to index (linear algebra vs GPU inference).
    No semantic generalisation ("happy" ≠ "joyful").
    Stronger privacy guarantee (zero external model weights).
    Zero installation beyond numpy.
    Genuinely useful for factual/keyword retrieval tasks.
"""

import numpy as np
import re
import time


# ==============================================================================
# FEATURE EXTRACTION (no pretrained vocabulary)
# ==============================================================================

def build_vocab(chunks, max_vocab=2000):
    """
    Build vocabulary from the document itself.
    No external word lists. Vocabulary = top N words in YOUR document.
    """
    counts = {}
    for chunk in chunks:
        for w in re.findall(r'\b[a-zA-Z]{2,}\b', chunk.lower()):
            counts[w] = counts.get(w, 0) + 1

    sorted_words = sorted(counts.items(), key=lambda x: -x[1])
    return {word: idx for idx, (word, _) in enumerate(sorted_words[:max_vocab])}


def tfidf_weights(chunks, vocab):
    """
    TF-IDF weights for vocabulary terms across chunks.
    Downweights common words automatically without a pretrained stopword list.
    """
    n = len(chunks)
    df = np.zeros(len(vocab))

    for chunk in chunks:
        words = set(re.findall(r'\b[a-zA-Z]{2,}\b', chunk.lower()))
        for w in words:
            if w in vocab:
                df[vocab[w]] += 1

    idf = np.log((n + 1) / (df + 1)) + 1  # smoothed IDF
    return idf


def bow_features(text, vocab, idf=None):
    """TF-IDF bag-of-words feature vector."""
    vec = np.zeros(len(vocab))
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    if not words:
        return vec
    for w in words:
        if w in vocab:
            vec[vocab[w]] += 1
    if idf is not None:
        vec *= idf
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def char_ngram_features(text, n=3, max_features=500):
    """
    Character n-gram features.
    Captures names, email addresses, technical terms, partial matches.
    Hash trick for fixed-size vector without a pretrained character vocab.
    """
    text_lower = text.lower()
    ngrams = {}
    for i in range(len(text_lower) - n + 1):
        gram = text_lower[i:i+n]
        if gram.strip():
            ngrams[gram] = ngrams.get(gram, 0) + 1

    vec = np.zeros(max_features)
    for gram, count in ngrams.items():
        idx = hash(gram) % max_features
        vec[idx] += count
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def encode_text(text, vocab, idf=None, char_features=500):
    """Combined BOW + character n-gram feature vector."""
    bow = bow_features(text, vocab, idf)
    char = char_ngram_features(text, max_features=char_features)
    return np.concatenate([bow, char])


# ==============================================================================
# DOCUMENT CHUNKING
# ==============================================================================

def chunk_document(text, chunk_size=400, overlap=50):
    """
    Split document into overlapping chunks.
    Respects email/section boundaries where possible.
    """
    # Try to split on email/section boundaries first
    sections = re.split(r'\n(?=From:|Subject:|Date:|---)', text)

    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) < 50:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            for i in range(0, len(section), chunk_size - overlap):
                chunk = section[i:i + chunk_size].strip()
                if len(chunk) > 50:
                    chunks.append(chunk)

    return chunks


# ==============================================================================
# SWEEP RETRIEVER
# ==============================================================================

class SweepRetriever:
    """
    Document retrieval via sweep reservoir.

    Zero pretrained models. Privacy-preserving. Runs on CPU with numpy only.

    The retriever builds a document-specific vocabulary and indexes
    chunks using random reservoir projections with directional sweep gates.
    Each chunk's embedding is context-aware: influenced by neighboring
    chunks in document order via the sweep gate.

    Parameters
    ----------
    stack_size : int
        Reservoir dimensionality. 512 is a good default.
    sigma : float
        Sweep gate width (in chunks). Larger = more context bleed.
        sigma=1.5 gives ~3 chunks of context influence.
    char_features : int
        Number of character n-gram hash features.

    Example
    -------
    >>> retriever = SweepRetriever()
    >>> with open("emails.txt") as f:
    ...     text = f.read()
    >>> retriever.build_index(text)
    >>> results = retriever.retrieve("who is Katrina", top_k=3)
    >>> for chunk, score in results:
    ...     print(f"[{score:.3f}] {chunk[:100]}")
    """

    def __init__(self, stack_size=512, sigma=1.5, char_features=500):
        self.stack_size = stack_size
        self.sigma = sigma
        self.char_features = char_features

        self.vocab = None
        self.idf = None
        self.W = None  # random projection matrix
        self.chunk_texts = []
        self.chunk_embeddings = None
        self.index_time = None

    def build_index(self, text_or_chunks, verbose=True):
        """
        Index a document. Trains the retriever on this specific document.
        Zero backpropagation. Pure linear algebra.

        Parameters
        ----------
        text_or_chunks : str or list of str
            Either raw document text (will be chunked automatically)
            or a pre-chunked list of strings.
        verbose : bool
            Print progress.

        Returns
        -------
        self (for chaining)
        """
        t0 = time.perf_counter()

        if isinstance(text_or_chunks, str):
            if verbose:
                print(f"  Chunking document ({len(text_or_chunks):,} chars)...")
            self.chunk_texts = chunk_document(text_or_chunks)
        else:
            self.chunk_texts = list(text_or_chunks)

        n = len(self.chunk_texts)
        if verbose:
            print(f"  Chunks: {n}")
            print(f"  Building document vocabulary...")

        self.vocab = build_vocab(self.chunk_texts)
        self.idf = tfidf_weights(self.chunk_texts, self.vocab)
        feat_dim = len(self.vocab) + self.char_features

        if verbose:
            print(f"  Vocabulary: {len(self.vocab)} words | Feature dim: {feat_dim}")

        # Random projection (fixed, never trained - the reservoir)
        rng = np.random.default_rng(42)
        self.W = rng.standard_normal((feat_dim, self.stack_size)) * np.sqrt(1.0 / feat_dim)

        if verbose:
            print(f"  Encoding {n} chunks...")
        raw_features = np.array([
            encode_text(c, self.vocab, self.idf, self.char_features)
            for c in self.chunk_texts
        ])

        if verbose:
            print(f"  Running sweep indexing...")
        self.chunk_embeddings = self._sweep_index(raw_features)

        self.index_time = time.perf_counter() - t0
        if verbose:
            print(f"  ✓ Index built in {self.index_time:.2f}s")
            print(f"  Index shape: {self.chunk_embeddings.shape}")

        return self

    def _sweep_index(self, features):
        """
        Apply sweep gate across chunks in document order.
        Creates context-aware chunk embeddings.

        features: (n_chunks, feat_dim)
        returns:  (n_chunks, 2 * stack_size)  [fwd + bwd]
        """
        n = features.shape[0]
        projected = np.tanh(features @ self.W)  # (n, stack_size)

        fwd = np.zeros((n, self.stack_size))
        bwd = np.zeros((n, self.stack_size))

        for k in range(n):
            pos = np.arange(n, dtype=float)

            # Forward gate: focus on chunk k
            g_fwd = np.exp(-0.5 * ((pos - k) / self.sigma) ** 2)
            g_fwd /= g_fwd.sum() + 1e-8
            fwd[k] = projected.T @ g_fwd

            # Backward gate: focus on mirror position
            mirror = n - 1 - k
            g_bwd = np.exp(-0.5 * ((pos - mirror) / self.sigma) ** 2)
            g_bwd /= g_bwd.sum() + 1e-8
            bwd[k] = projected.T @ g_bwd

        combined = np.concatenate([fwd, bwd], axis=1)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        return combined / (norms + 1e-8)

    def retrieve(self, query, top_k=5):
        """
        Find top_k most relevant chunks for a query.

        Parameters
        ----------
        query : str
            Natural language query.
        top_k : int
            Number of results to return.

        Returns
        -------
        results : list of (str, float)
            List of (chunk_text, similarity_score) sorted by relevance.
        """
        if self.vocab is None or self.chunk_embeddings is None:
            raise RuntimeError("Call build_index() before retrieve().")

        q_feat = encode_text(query, self.vocab, self.idf, self.char_features)
        q_proj = np.tanh(q_feat @ self.W)
        q_norm = np.linalg.norm(q_proj)
        q_proj = q_proj / (q_norm + 1e-8)

        # Compare against forward sweep half (natural reading order)
        fwd_embeddings = self.chunk_embeddings[:, :self.stack_size]
        sims = fwd_embeddings @ q_proj

        top_idx = sims.argsort()[-top_k:][::-1]
        return [(self.chunk_texts[i], float(sims[i])) for i in top_idx]

    def retrieve_with_context(self, query, top_k=3, context_window=1):
        """
        Retrieve top chunks and include surrounding context chunks.

        Parameters
        ----------
        query : str
        top_k : int
        context_window : int
            Number of neighboring chunks to include on each side.

        Returns
        -------
        results : list of (str, float)
            Expanded results with context neighbors merged.
        """
        hits = self.retrieve(query, top_k=top_k)
        expanded = []
        seen = set()

        for chunk, score in hits:
            idx = self.chunk_texts.index(chunk)
            start = max(0, idx - context_window)
            end = min(len(self.chunk_texts), idx + context_window + 1)

            merged = "\n".join(self.chunk_texts[start:end])
            key = (start, end)
            if key not in seen:
                seen.add(key)
                expanded.append((merged, score))

        return expanded
