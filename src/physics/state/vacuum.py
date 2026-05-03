"""Vacuum quantum state.

This module defines the vacuum state |0⟩ of a bosonic mode.
"""

from qutip import basis, Qobj
from .base import QuantumState


class VacuumState(QuantumState):
    """Vacuum quantum state.

    The vacuum state |0⟩ is the ground state of a bosonic mode,
    containing zero excitations.

    It is defined as:

        |0⟩

    Args:
        cutoff (int): Hilbert space dimension.

    Raises:
        ValueError: If cutoff is non-positive.
    """

    def __init__(self, cutoff: int) -> None:
        """Initialize a vacuum state.

        Args:
            cutoff (int): Hilbert space dimension.
        """
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")

        self.cutoff = cutoff

    def ket(self) -> Qobj:
        """Return the vacuum state vector.

        Returns:
            Qobj: Ket vector |0⟩.
        """
        return basis(self.cutoff, 0)
