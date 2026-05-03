"""Photon loss (amplitude damping) quantum channel.

This module implements a bosonic loss channel using the Lindblad
master equation formalism.
"""

import numpy as np
from qutip import destroy, mesolve, Qobj
from .base import QuantumChannel


class LossChannel(QuantumChannel):
    """Photon loss (amplitude damping) channel.

    Models energy dissipation in a bosonic mode due to photon loss,
    described by a Lindblad master equation with a single collapse
    operator proportional to the annihilation operator.

    The evolution of the density matrix ρ is given by:

        dρ/dt = γ (a ρ a† - 1/2 {a† a, ρ})

    where a is the annihilation operator and γ is the loss rate.

    Physically, this corresponds to coupling the mode to a
    zero-temperature bath, leading to exponential decay of excitations.

    In this implementation, the channel is obtained by solving the
    master equation over a fixed time interval t ∈ [0, 1], so γ
    effectively sets the strength of the loss.

    Args:
        cutoff (int): Hilbert space dimension.
        gamma (float): Loss rate (decay strength).

    Raises:
        ValueError: If gamma is negative.
    """

    def __init__(self, cutoff: int, gamma: float) -> None:
        """Initialize the loss channel.

        Args:
            cutoff (int): Hilbert space dimension.
            gamma (float): Loss rate.

        Raises:
            ValueError: If gamma is negative.
        """
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")

        self.cutoff = cutoff
        self.gamma = gamma

    def apply(self, rho: Qobj) -> Qobj:
        """Apply the loss channel to a density matrix.

        Evolves the input state according to the Lindblad master equation
        with a single collapse operator √γ · a over the time interval [0, 1].

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
            c_ops=[np.sqrt(self.gamma) * a],
        )

        return result.states[-1]