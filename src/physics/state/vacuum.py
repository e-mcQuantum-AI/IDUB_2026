from qutip import basis
from src.physics.state.base import QuantumState

class VacuumState(QuantumState):

    def __init__(self, cutoff: int):
        self.cutoff = cutoff

    def ket(self):
        return basis(self.cutoff, 0)
