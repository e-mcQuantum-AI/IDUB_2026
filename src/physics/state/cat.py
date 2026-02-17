from qutip import coherent
from base import QuantumState

class CatState(QuantumState):

    def __init__(self, alpha: complex, cutoff: int):
        self.alpha = alpha
        self.cutoff = cutoff

    def ket(self):
        psi = coherent(self.cutoff, self.alpha) \
            + coherent(self.cutoff, -self.alpha)
        return psi.unit()
