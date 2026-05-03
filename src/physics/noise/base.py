"""Abstract base class for quantum channels.

This module defines the interface for quantum channels acting on
density matrices in a bosonic Hilbert space.
"""

from abc import ABC, abstractmethod
from qutip import Qobj


class QuantumChannel(ABC):
    """Abstract base class for quantum channels.

    A quantum channel represents a completely positive, trace-preserving
    (CPTP) map acting on a density matrix.

    Subclasses must implement the ``apply`` method, which defines how
    the channel transforms an input state.

    """

    @abstractmethod
    def apply(self, rho: Qobj) -> Qobj:
        """Apply the quantum channel to a density matrix.

        Args:
            rho (Qobj): Input density matrix.

        Returns:
            Qobj: Output density matrix after applying the channel.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        pass
