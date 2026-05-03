"""Fock (number) quantum state.

This module defines a Fock state (number state) in a truncated
bosonic Hilbert space.
"""

from qutip import basis, Qobj
from .base import QuantumState


class FockState(QuantumState):
    """Fock (number) quantum state.

    A Fock state |n⟩ is a quantum state with a well-defined number
    of excitations in a bosonic mode.

    It is defined as:

        |n⟩

    where n is the occupation number.

    Args:
        n (int): Photon number (excitation level).
        cutoff (int): Hilbert space dimension.

    Raises:
        ValueError: If n is outside valid range or cutoff is invalid.
    """

    def __init__(self, n: int, cutoff: int) -> None:
        """Initialize a Fock state.

        Args:
            n (int): Photon number.
            cutoff (int): Hilbert space dimension.
        """
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if not (0 <= n < cutoff):
            raise ValueError("n must satisfy 0 <= n < cutoff.")

        self.n = n
        self.cutoff = cutoff

    def ket(self) -> Qobj:
        """Return the Fock state vector.

        Constructs the number state |n⟩ in a truncated Fock basis.

        Returns:
            Qobj: Ket vector representing the Fock state.
        """
        return basis(self.cutoff, self.n)
