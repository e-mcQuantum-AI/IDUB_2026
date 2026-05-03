"""Schrödinger cat state (bosonic superposition state).

This module defines a quantum superposition of coherent states,
commonly known as a Schrödinger cat state.
"""

from qutip import coherent, Qobj
from .base import QuantumState


class CatState(QuantumState):
    """Schrödinger cat state.

    A cat state is a quantum superposition of two coherent states
    with opposite amplitudes:

        |ψ⟩ ∝ |α⟩ + |−α⟩

    where |α⟩ is a coherent state.

    These states exhibit strong quantum interference effects and are
    widely used in bosonic quantum error correction and quantum optics.

    Args:
        alpha (complex): Coherent state amplitude.
        cutoff (int): Hilbert space dimension (Fock truncation).

    Raises:
        ValueError: If cutoff is non-positive.
    """

    def __init__(self, alpha: complex, cutoff: int) -> None:
        """Initialize a cat state.

        Args:
            alpha (complex): Coherent amplitude.
            cutoff (int): Hilbert space dimension.
        """
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")

        self.alpha = alpha
        self.cutoff = cutoff

    def ket(self) -> Qobj:
        """Return the normalized cat state vector.

        Constructs a superposition of two coherent states:

            |ψ⟩ = |α⟩ + |−α⟩

        and normalizes the result.

        Returns:
            Qobj: Normalized ket vector representing the cat state.
        """
        psi: Qobj = coherent(self.cutoff, self.alpha) \
            + coherent(self.cutoff, -self.alpha)
        return psi.unit()
