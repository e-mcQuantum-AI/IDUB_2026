"""Gaussian quadrature noise quantum channel.

This module implements a continuous-variable Gaussian noise channel (quadrature
diffusion) using the Lindblad master equation formalism.
"""

import numpy as np
from qutip import destroy, mesolve, Qobj
from .base import QuantumChannel


class GaussianNoiseChannel(QuantumChannel):
    """Gaussian quadrature noise (diffusion) channel.

    Models the addition of classical Gaussian noise to the position (X) and
    momentum (P) quadratures of a bosonic mode. This process is described by
    a Lindblad master equation with two collapse operators proportional to the
    quadrature operators.

    The evolution of the density matrix ρ is given by:

        dρ/dt = γ (X ρ X - 1/2 {X², ρ}) + γ (P ρ P - 1/2 {P², ρ})

    where X = (a + a†)/√2 and P = (a - a†)/(i√2) are the quadrature operators,
    and γ represents the diffusion rate (noise strength).

    Physically, this corresponds to random, fluctuating displacements in the
    phase space, which symmetrically increases the variance of both quadratures
    and adds thermal-like excitations to the system.

    In this implementation, the channel is obtained by solving the
    master equation over a fixed time interval t ∈ [0, 1], so γ
    effectively sets the variance of the added noise.

    Args:
        cutoff (int): Hilbert space dimension.
        gamma (float): Diffusion rate (noise strength).

    Raises:
        ValueError: If gamma is negative.
    """

    def __init__(self, cutoff: int, gamma: float) -> None:
        """Initialize the Gaussian noise channel.

        Args:
            cutoff (int): Hilbert space dimension.
            gamma (float): Diffusion rate.

        Raises:
            ValueError: If gamma is negative.
        """
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        self.cutoff = cutoff
        self.gamma = gamma

    def apply(self, rho: Qobj) -> Qobj:
        """Apply the Gaussian noise channel to a density matrix.

        Evolves the input state according to the quadrature diffusion Lindblad
        master equation with collapse operators √γ · X and √γ · P over the
        time interval [0, 1].

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

        X = (a + a.dag()) / np.sqrt(2)
        P = (a - a.dag()) / (1j * np.sqrt(2))

        c_ops = [
            np.sqrt(self.gamma) * X,
            np.sqrt(self.gamma) * P
        ]

        result = mesolve(
            H=0 * a.dag() * a,
            rho0=rho,
            tlist=[0, 1],
            c_ops=c_ops,
        )

        return result.states[-1]
