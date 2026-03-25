class MeasurementPipeline:

    def __init__(self, measurement, noise=None):
        self.measurement = measurement
        self.noise = noise

    def run(self, state):
        rho = state.density_matrix()

        if self.noise:
            rho = self.noise.apply(rho)

        return self.measurement.measure(rho)
