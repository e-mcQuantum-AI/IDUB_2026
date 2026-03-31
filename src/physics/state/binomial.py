import numpy as np
from qutip import Qobj
from scipy.special import comb
from .base import QuantumState


class BinomialState(QuantumState):

    def __init__(self, N: int, p: float, cutoff: int):
        self.N = N
        self.p = p
        self.cutoff = cutoff

    def ket(self):
        vec = np.zeros(self.cutoff, dtype=complex)

        for n in range(self.N + 1):
            coeff = np.sqrt(comb(self.N, n)) \
                    * (self.p**(n/2)) \
                    * ((1-self.p)**((self.N-n)/2))
            vec[n] = coeff

        return Qobj(vec).unit()
