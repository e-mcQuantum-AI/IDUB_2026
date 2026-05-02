import numpy as np
from qutip import wigner, Qobj


class WignerMeasurement:

    def __init__(self, x_max=5, resolution=64):
        self.xvec = np.linspace(-x_max, x_max, resolution)

    def measure(self, rho: Qobj):
        W = wigner(rho, self.xvec, self.xvec)
        return W
