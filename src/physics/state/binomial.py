import numpy as np
from qutip import Qobj
from scipy.special import comb
from .base import QuantumState


class BinomialState(QuantumState):
    """ Binomial bosonic state.

    Binomial states are finite superpositions of Fock states with
    coefficients following a binomial distribution. They form a
    family of bosonic code states that can approximately protect
    against photon loss and dephasing errors by encoding quantum
    information in a truncated Hilbert space.

    The state is defined as:

        |ψ> = sum_{n=0}^N sqrt(C(N, n)) * p^{n/2} * (1 - p)^{(N-n)/2} |n>

        where C(N, n) is the binomial coefficient, N is the maximum
        excitation number, and p ∈ [0, 1] controls the distribution.

    Reference:
        M. H. Michael et al.,
        "New Class of Quantum Error-Correcting Codes for a Bosonic Mode",
        Phys. Rev. X 6, 031006 (2016).
        https://doi.org/10.1103/PhysRevX.6.031006

    Args:
        N: Maximum Fock number in the superposition.
        p: Probability parameter of the binomial distribution.
        cutoff: Hilbert space dimension.
    """

    def __init__(self, N: int, p: float, cutoff: int) -> None:
        self.N = N
        self.p = p
        self.cutoff = cutoff

    def ket(self) -> Qobj:
        vec = np.zeros(self.cutoff, dtype=complex)

        for n in range(self.N + 1):
            coeff = np.sqrt(comb(self.N, n)) \
                    * (self.p**(n/2)) \
                    * ((1-self.p)**((self.N-n)/2))
            vec[n] = coeff

        return Qobj(vec).unit()
