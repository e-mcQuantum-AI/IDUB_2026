from qutip import coherent, Qobj
from .base import QuantumState

class CoherentState(QuantumState):

    def __init__(self, alpha: complex, cutoff: int) -> None:
        self.alpha = alpha
        self.cutoff = cutoff

    def ket(self) -> Qobj:
        return coherent(self.cutoff, self.alpha)
