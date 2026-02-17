import numpy as np

class DatasetGenerator:

    def __init__(self, state_sampler, pipeline):
        self.state_sampler = state_sampler
        self.pipeline = pipeline

    def generate(self, n_samples):
        X = []
        y = []

        for _ in range(n_samples):
            state, label = self.state_sampler()
            W = self.pipeline.run(state)

            X.append(W)
            y.append(label)

        return np.array(X), np.array(y)
