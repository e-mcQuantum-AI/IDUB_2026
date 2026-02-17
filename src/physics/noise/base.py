from abc import ABC, abstractmethod
from qutip import Qobj

class QuantumChannel(ABC):

    @abstractmethod
    def apply(self, rho: Qobj) -> Qobj:
        pass
