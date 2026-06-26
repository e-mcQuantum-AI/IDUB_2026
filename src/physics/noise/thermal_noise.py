"""Thermal noise (finite temperature bath) quantum channel.

This module implements a bosonic thermal noise channel using the Lindblad
master equation formalism with a thermal environment.
"""

import numpy as np
from qutip import destroy, mesolve, Qobj
from .base import QuantumChannel


class ThermalNoiseChannel(QuantumChannel):
    """Thermal noise channel (interaction with a finite-temperature bath).

    Models a bosonic mode coupled to a thermal reservoir with a mean photon
    number n_th. The evolution is governed by a Lindblad master equation with
    two collapse operators: one for photon emission into the bath, and one for
    photon absorption from the bath.

    The evolution of the density matrix ρ is given by:

        dρ/dt = γ(1 + n_th) (a ρ a† - 1/2 {a† a, ρ}) + γ n_th (a† ρ a - 1/2 {a a†, ρ})

    where a is the annihilation operator, γ is the coupling (decay) rate,
    and n_th is the mean thermal photon number of the bath.

    Physically, if the system is left in this channel for a long time (t → ∞),
    it will eventually relax to a thermal state with mean photon number n_th,
    regardless of its initial state.

    Args:
        cutoff (int): Hilbert space dimension.
        gamma (float): Coupling rate to the reservoir.
        n_th (float): Mean thermal photon number of the bath.

    Raises:
        ValueError: If gamma or n_th is negative.
    """

    def __init__(self, cutoff: int, gamma: float, n_th: float) -> None:
        """Initialize the thermal noise channel.

        Args:
            cutoff (int): Hilbert space dimension.
            gamma (float): Coupling rate.
            n_th (float): Mean thermal photon number.

        Raises:
            ValueError: If gamma or n_th is negative.
        """
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if n_th < 0:
            raise ValueError(f"n_th must be non-negative, got {n_th}")

        self.cutoff = cutoff
        self.gamma = gamma
        self.n_th = n_th

    def apply(self, rho: Qobj) -> Qobj:
        """Apply the thermal noise channel to a density matrix.

        Evolves the input state according to the Lindblad master equation
        with collapse operators √[γ(1+n_th)] · a and √[γ · n_th] · a† over
        the time interval [0, 1].

        Args:
            rho (Qobj): Input density matrix.

        Returns:
            Qobj: Output density matrix after thermal evolution.

        Raises:
            ValueError: If the input is not an operator (density matrix).
        """
        if not rho.isoper:
            raise ValueError("Expected a density matrix (Qobj operator).")

        a: Qobj = destroy(self.cutoff)

        c_ops = [
            np.sqrt(self.gamma * (1.0 + self.n_th)) * a,
            np.sqrt(self.gamma * self.n_th) * a.dag()
        ]

        result = mesolve(
            H=0 * a.dag() * a,
            rho0=rho,
            tlist=[0, 1],
            c_ops=c_ops,
        )

        return result.states[-1]
