"""
Sequence task utilities and benchmarking.

Standard tasks used to validate the sweep reservoir:
    Copy:    output[i] = input[i]
    Reverse: output[i] = input[N-1-i]
    Shift:   output[i] = input[i-1] (circular)

Validated results (sweep reservoir vs Transformer):
    Copy:    ~100% vs 100%  at 64x training speed
    Reverse: ~100% vs 100%  at 64x training speed
    Shift:    ~52% vs 100%  (sweep has weakness here - needs locality)
"""

import numpy as np
import time


def generate_sequences(task, vocab_size, seq_len, n):
    """
    Generate sequence task data.

    Parameters
    ----------
    task : str
        'copy', 'reverse', or 'shift'
    vocab_size : int
    seq_len : int
    n : int
        Number of sequences.

    Returns
    -------
    sequences, targets : list of ndarray
    """
    sequences, targets = [], []
    for _ in range(n):
        s = np.random.randint(0, vocab_size, seq_len)
        if task == 'copy':
            t = s.copy()
        elif task == 'reverse':
            t = s[::-1].copy()
        elif task == 'shift':
            t = np.roll(s, 1)
        else:
            raise ValueError(f"Unknown task: {task}. Use 'copy', 'reverse', or 'shift'.")
        sequences.append(s)
        targets.append(t)
    return sequences, targets


def solve_sequence_task(reservoir, task, vocab_size=8, seq_len=6,
                         n_train=600, reg=0.1):
    """
    Train a sweep reservoir readout on a sequence task.

    Parameters
    ----------
    reservoir : SweepReservoir
    task : str
        'copy', 'reverse', or 'shift'
    vocab_size : int
    seq_len : int
    n_train : int
    reg : float
        Ridge regularisation.

    Returns
    -------
    W_out : ndarray
        Trained readout weights.
    train_time_ms : float
        Training time in milliseconds.
    """
    seqs, tgts = generate_sequences(task, vocab_size, seq_len, n_train)
    t0 = time.perf_counter()
    W_out = reservoir.train(seqs, tgts, reg=reg)
    train_time_ms = (time.perf_counter() - t0) * 1000
    return W_out, train_time_ms


def evaluate_sequence_task(reservoir, W_out, task, vocab_size=8,
                            seq_len=6, n_test=200):
    """
    Evaluate a trained readout on a sequence task.

    Parameters
    ----------
    reservoir : SweepReservoir
    W_out : ndarray
    task : str
    vocab_size : int
    seq_len : int
    n_test : int

    Returns
    -------
    accuracy : float in [0, 1]
    """
    seqs, tgts = generate_sequences(task, vocab_size, seq_len, n_test)
    return reservoir.accuracy(seqs, tgts, W_out)


def benchmark(reservoir, tasks=('copy', 'reverse', 'shift'),
              vocab_size=8, seq_len=6, n_train=600, n_test=200,
              verbose=True):
    """
    Run full benchmark across multiple tasks.

    Parameters
    ----------
    reservoir : SweepReservoir
    tasks : tuple of str
    vocab_size : int
    seq_len : int
    n_train : int
    n_test : int
    verbose : bool

    Returns
    -------
    results : dict
        {task: {'accuracy': float, 'train_ms': float}}
    """
    results = {}

    for task in tasks:
        W_out, train_ms = solve_sequence_task(
            reservoir, task, vocab_size, seq_len, n_train
        )
        acc = evaluate_sequence_task(
            reservoir, W_out, task, vocab_size, seq_len, n_test
        )
        results[task] = {'accuracy': acc, 'train_ms': train_ms}

        if verbose:
            print(f"  {task:<10} acc={acc:.1%}  train={train_ms:.0f}ms")

    return results
