"""Mixture quantum channel.

This module implements a simple probabilistic mixture channel that
combines an input state with a fixed reference state.
"""

from qutip import Qobj
from .base import QuantumChannel


class MixtureChannel(QuantumChannel):
    """Convex mixture quantum channel.

    This channel performs a probabilistic mixture between the input
    density matrix ρ and a fixed reference state ρ_other:

        ρ_out = p ρ + (1 - p) ρ_other

    This is not a physical dynamical channel in general, but can be used
    to model classical noise, state replacement, or imperfect preparation.

    Args:
        p (float): Mixing probability in [0, 1].
        rho_other (Qobj): Reference density matrix used in the mixture.

    Raises:
        ValueError: If p is not in [0, 1].
    """

    def __init__(self, p: float, rho_other: Qobj) -> None:
        """Initialize the mixture channel.

        Args:
            p (float): Mixing probability in [0, 1].
            rho_other (Qobj): Reference density matrix.
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")

        self.p = p
        self.rho_other = rho_other

    def apply(self, rho: Qobj) -> Qobj:
        """Apply the mixture channel.

        Computes a convex combination of the input state and a fixed
        reference state.

        Args:
            rho (Qobj): Input density matrix.

        Returns:
            Qobj: Mixed density matrix.
        """
        return self.p * rho + (1 - self.p) * self.rho_other
