from qutip import coherent, Qobj
from .base import QuantumState

class CatState(QuantumState):

    def __init__(self, alpha: complex, cutoff: int) ->  None:
        self.alpha = alpha
        self.cutoff = cutoff

    def ket(self) -> Qobj:
        psi: Qobj = coherent(self.cutoff, self.alpha) \
            + coherent(self.cutoff, -self.alpha)
        return psi.unit()
