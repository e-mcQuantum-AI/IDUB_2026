"""Binomial bosonic quantum state.

This module implements binomial bosonic code states used in quantum
error correction for bosonic modes.
"""

import numpy as np
from qutip import Qobj
from scipy.special import comb
from .base import QuantumState


class BinomialState(QuantumState):
    """Binomial bosonic state.

    Binomial states are finite superpositions of Fock states with
    coefficients following a binomial distribution. They form a family
    of bosonic code states that can approximately protect against photon
    loss and dephasing errors by encoding quantum information in a
    truncated Hilbert space.

    The state is defined as:

        |ψ⟩ = ∑_{n=0}^N √C(N, n) · p^{n/2} · (1 - p)^{(N-n)/2} |n⟩

    where C(N, n) is the binomial coefficient, N is the maximum excitation
    number, and p ∈ [0, 1] controls the distribution.

    References:
        Michael, M. H. et al. (2016).
        "New Class of Quantum Error-Correcting Codes for a Bosonic Mode".
        Phys. Rev. X 6, 031006.
        https://doi.org/10.1103/PhysRevX.6.031006

    Args:
        N (int): Maximum Fock number in the superposition.
        p (float): Probability parameter of the binomial distribution.
        cutoff (int): Hilbert space dimension.

    Raises:
        ValueError: If p is not in [0, 1] or N >= cutoff.
    """

    def __init__(self, N: int, p: float, cutoff: int) -> None:
        """Initialize a binomial state.

        Args:
            N (int): Maximum Fock number.
            p (float): Probability parameter.
            cutoff (int): Hilbert space dimension.
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")
        if N >= cutoff:
            raise ValueError("N must be smaller than cutoff.")

        self.N = N
        self.p = p
        self.cutoff = cutoff

    def ket(self) -> Qobj:
        """Return the state vector (ket) representation.

        Constructs the binomial superposition in the Fock basis and
        normalizes the resulting vector.

        Returns:
            Qobj: Normalized ket vector representing the state.
        """
        vec = np.zeros(self.cutoff, dtype=complex)

        for n in range(self.N + 1):
            coeff = (
                np.sqrt(comb(self.N, n))
                * (self.p ** (n / 2))
                * ((1 - self.p) ** ((self.N - n) / 2))
            )
            vec[n] = coeff

        return Qobj(vec).unit()
