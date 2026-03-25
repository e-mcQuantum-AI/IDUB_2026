from qutip import thermal_dm
from qutip import Qobj
from .base import QuantumState

class ThermalState(QuantumState):

    def __init__(self, n_th: float, cutoff: int):
        self.n_th = n_th
        self.cutoff = cutoff

    def ket(self):
        raise NotImplementedError("Thermal state is mixed.")

    def density_matrix(self) -> Qobj:
        return thermal_dm(self.cutoff, self.n_th)
