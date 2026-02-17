from abc import ABC, abstractmethod
from qutip import Qobj

class QuantumState(ABC):

    @abstractmethod
    def ket(self) -> Qobj:
        pass

    def density_matrix(self) -> Qobj:
        psi = self.ket()
        return psi * psi.dag()
