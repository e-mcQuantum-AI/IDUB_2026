"""Finite-dimensional bosonic Hilbert space utilities.

This module provides creation of standard operators (annihilation,
creation, identity) in a truncated Fock space representation.
"""

from dataclasses import dataclass
from qutip import destroy, create, qeye, Qobj


@dataclass
class HilbertSpace:
    """Finite-dimensional Hilbert space for a bosonic mode.

    This class provides standard operators in a truncated Fock basis:

        a      — annihilation operator
        a†     — creation operator
        I      — identity operator

    The Hilbert space is truncated at a finite cutoff dimension.

    Args:
        cutoff (int): Dimension of the truncated Fock space.

    Raises:
        ValueError: If cutoff is not positive.
    """

    cutoff: int

    def __post_init__(self) -> None:
        """Validate Hilbert space parameters."""
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive.")

    def a(self) -> Qobj:
        """Return annihilation operator a.

        Returns:
            Qobj: Annihilation operator in truncated Fock basis.
        """
        return destroy(self.cutoff)

    def adag(self) -> Qobj:
        """Return creation operator a†.

        Returns:
            Qobj: Creation operator in truncated Fock basis.
        """
        return create(self.cutoff)

    def identity(self) -> Qobj:
        """Return identity operator.

        Returns:
            Qobj: Identity operator in truncated Hilbert space.
        """
        return qeye(self.cutoff)
