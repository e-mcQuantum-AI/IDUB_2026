from qutip import destroy, mesolve
import numpy as np

class LossChannel:

    def __init__(self, cutoff: int, gamma: float):
        self.cutoff = cutoff
        self.gamma = gamma

    def apply(self, rho):
        a = destroy(self.cutoff)
        result = mesolve(
            H=0 * a.dag() * a,
            rho0=rho,
            tlist=[0, 1],
            c_ops=[np.sqrt(self.gamma) * a]
        )
        return result.states[-1]
