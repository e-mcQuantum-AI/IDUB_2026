"""Coherent quantum state.

This module defines a coherent state of a bosonic mode.
"""

from qutip import coherent, Qobj
from .base import QuantumState


class CoherentState(QuantumState):
    """Coherent quantum state.

    A coherent state |α⟩ is an eigenstate of the annihilation operator
    and represents the closest quantum analogue to a classical harmonic
    oscillator state.

    It is defined as:

        |α⟩

    where α ∈ ℂ is the displacement amplitude.

    Args:
        alpha (complex): Coherent state amplitude.
        cutoff (int): Hilbert space dimension (Fock truncation).

    Raises:
        ValueError: If cutoff is non-positive.
    """

    def __init__(self, alpha: complex, cutoff: int) -> None:
        """Initialize a coherent state.

        Args:
            alpha (complex): Coherent amplitude.
            cutoff (int): Hilbert space dimension.
        """
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")

        self.alpha = alpha
        self.cutoff = cutoff

    def ket(self) -> Qobj:
        """Return the coherent state vector.

        Constructs the coherent state |α⟩ in a truncated Fock basis.

        Returns:
            Qobj: Ket vector representing the coherent state.
        """
        return coherent(self.cutoff, self.alpha)
