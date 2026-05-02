from qutip import basis, Qobj
from .base import QuantumState

class FockState(QuantumState):

    def __init__(self, n: int, cutoff: int) -> None:
        self.n = n
        self.cutoff = cutoff

    def ket(self) -> Qobj:
        return basis(self.cutoff, self.n)
