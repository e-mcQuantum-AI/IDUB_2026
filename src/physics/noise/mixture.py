class MixtureChannel:

    def __init__(self, p: float, rho_other):
        self.p = p
        self.rho_other = rho_other

    def apply(self, rho):
        return self.p * rho + (1 - self.p) * self.rho_other
