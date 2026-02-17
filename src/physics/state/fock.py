from qutip import basis
from base import QuantumState

class FockState(QuantumState):

    def __init__(self, n: int, cutoff: int):
        self.n = n
        self.cutoff = cutoff

    def ket(self):
        return basis(self.cutoff, self.n)
