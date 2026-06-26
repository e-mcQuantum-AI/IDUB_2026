"""Quantum amplification channel.

This module implements a bosonic linear amplification channel using the Lindblad
master equation formalism.
"""

import numpy as np
from qutip import destroy, mesolve, Qobj
from .base import QuantumChannel


class AmplificationChannel(QuantumChannel):
    """Quantum amplification channel.

    Models energy amplification in a bosonic mode, described by a Lindblad
    master equation with a single collapse operator proportional to the
    creation operator.

    The evolution of the density matrix ρ is given by:

        dρ/dt = γ (a† ρ a - 1/2 {a a†, ρ})

    where a† is the creation operator and γ is the amplification rate.

    Physically, this corresponds to coupling the mode to an inverted population
    bath (active medium), which adds photons to the system while inevitably
    introducing quantum noise (spontaneous emission).

    In this implementation, the channel is obtained by solving the
    master equation over a fixed time interval t ∈ [0, 1], so γ
    effectively sets the strength of the amplification.

    Args:
        cutoff (int): Hilbert space dimension.
        gamma (float): Amplification rate (gain strength).

    Raises:
        ValueError: If gamma is negative.
    """

    def __init__(self, cutoff: int, gamma: float) -> None:
        """Initialize the amplification channel.

        Args:
            cutoff (int): Hilbert space dimension.
            gamma (float): Amplification rate.

        Raises:
            ValueError: If gamma is negative.
        """
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        self.cutoff = cutoff
        self.gamma = gamma

    def apply(self, rho: Qobj) -> Qobj:
        """Apply the amplification channel to a density matrix.

        Evolves the input state according to the Lindblad master equation
        with a single collapse operator √γ · a† over the time interval [0, 1].

        Args:
            rho (Qobj): Input density matrix.

        Returns:
            Qobj: Output density matrix after evolution.

        Raises:
            ValueError: If the input is not an operator (density matrix).
        """
        if not rho.isoper:
            raise ValueError("Expected a density matrix (Qobj operator).")

        a: Qobj = destroy(self.cutoff)

        result = mesolve(
            H=0 * a.dag() * a,
            rho0=rho,
            tlist=[0, 1],
            c_ops=[np.sqrt(self.gamma) * a.dag()],
        )

        return result.states[-1]
