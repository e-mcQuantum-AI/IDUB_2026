"""Abstract base class for quantum states.

This module defines the interface for pure quantum states and provides
a default implementation for constructing density matrices.
"""

from abc import ABC, abstractmethod
from qutip import Qobj


class QuantumState(ABC):
    """Abstract base class for quantum states.

    A quantum state is represented via its ket vector in Hilbert space.
    Subclasses must implement the ``ket`` method.

    The class also provides a default method for constructing the
    corresponding density matrix:

        ρ = |ψ⟩⟨ψ|

    """

    @abstractmethod
    def ket(self) -> Qobj:
        """Return the state vector (ket) representation.

        Returns:
            Qobj: Pure state vector |ψ⟩.
        """
        pass

    def density_matrix(self) -> Qobj:
        """Construct the density matrix of the quantum state.

        The density matrix is computed from the ket vector as:

            ρ = |ψ⟩⟨ψ|

        Returns:
            Qobj: Density matrix representation of the state.
        """
        psi: Qobj = self.ket()
        return psi * psi.dag()
