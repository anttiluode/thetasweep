"""
ThetaSweep
==========
Biologically-grounded sequence processing via directional reservoir sweeps.

Inspired by:
- Vollan et al. (2025): Theta phase sweeps in entorhinal-hippocampal circuits
- Empirical observation of box attractor at theta frequency in frontal EEG
  (PerceptionLab phase space node, A. Luode 2024)
- Phi-Dwell eigenmode analysis of EEG (Luode 2024)

Core insight:
    The brain does not store sequences as static feature vectors.
    It traverses them rhythmically via a theta-frequency clock.
    Position is encoded in WHEN a neuron fires (phase),
    not HOW MUCH it fires (amplitude).

    A box attractor in frontal phase space at theta frequency =
    a 4-state limit cycle clock driving bidirectional sequence sweeps.

    This library implements that mechanism computationally:
    - Random reservoir (high-dimensional projection)
    - Directional sweep gate (phase precession analogue)
    - Ridge regression readout (instantaneous, zero backprop)

Author: Antti Luode
License: ThetaSweep Source Available License v1.0
         Free for personal/research use.
         Commercial use requires a license from the author.
         Contact: [your contact]
"""

from .reservoir import SweepReservoir
from .retriever import SweepRetriever
from .tasks import solve_sequence_task, evaluate_sequence_task

__version__ = "0.1.0"
__author__ = "Antti Luode"
__license__ = "ThetaSweep Source Available License v1.0"

__all__ = [
    "SweepReservoir",
    "SweepRetriever",
    "solve_sequence_task",
    "evaluate_sequence_task",
]
