from qutip import coherent
from .base import QuantumState

class CoherentState(QuantumState):

    def __init__(self, alpha: complex, cutoff: int):
        self.alpha = alpha
        self.cutoff = cutoff

    def ket(self):
        return coherent(self.cutoff, self.alpha)
