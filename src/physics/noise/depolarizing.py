"""Depolarizing quantum channel.

This module implements a bosonic depolarizing channel that replaces the input
state with a maximally mixed state with a given probability.
"""

from qutip import qeye, Qobj
from .base import QuantumChannel


class DepolarizingChannel(QuantumChannel):
    """Depolarizing channel.

    Models a process where the input quantum state ρ is preserved with
    probability (1 - p), and is completely replaced by a white noise
    (the maximally mixed state I/d) with probability p.

    The transformation of the density matrix ρ is given by:

        ρ → (1 - p)ρ + p · (I / d)

    where I is the identity operator and d is the dimension of the
    Hilbert space (cutoff).

    Physically, this corresponds to a worst-case noise scenario where information
    is completely erased and replaced by a random isotropic mixture, mimicking
    a detector that occasionally yields a completely random result.

    Args:
        cutoff (int): Hilbert space dimension (d).
        p (float): Depolarization probability, must be in the range [0, 1].

    Raises:
        ValueError: If p is not in the range [0, 1].
    """

    def __init__(self, cutoff: int, p: float) -> None:
        """Initialize the depolarizing channel.

        Args:
            cutoff (int): Hilbert space dimension.
            p (float): Depolarization probability.

        Raises:
            ValueError: If p is not in the range [0, 1].
        """
        if p < 0 or p > 1:
            raise ValueError(f"Probability p must be in the range [0, 1], got {p}")

        self.cutoff = cutoff
        self.p = p

    def apply(self, rho: Qobj) -> Qobj:
        """Apply the depolarizing channel to a density matrix.

        Mixes the input state with the maximally mixed state according to the
        depolarization probability p.

        Args:
            rho (Qobj): Input density matrix.

        Returns:
            Qobj: Output density matrix after depolarization.

        Raises:
            ValueError: If the input is not an operator (density matrix).
        """
        if not rho.isoper:
            raise ValueError("Expected a density matrix (Qobj operator).")

        d = self.cutoff
        identity = qeye(d)
        maximally_mixed = identity / d

        return (1.0 - self.p) * rho + self.p * maximally_mixed
