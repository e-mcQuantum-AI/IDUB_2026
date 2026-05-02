from qutip import basis, Qobj
from .base import QuantumState

class VacuumState(QuantumState):

    def __init__(self, cutoff: int) -> None:
        self.cutoff = cutoff

    def ket(self) -> Qobj:
        return basis(self.cutoff, 0)
