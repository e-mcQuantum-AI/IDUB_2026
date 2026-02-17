import random
from src.physics.state.vacuum import VacuumState
from src.physics.state.coherent import CoherentState

def simple_sampler(cutoff):

    choice = random.choice(["vacuum", "coherent"])

    if choice == "vacuum":
        return VacuumState(cutoff), 0

    else:
        alpha = random.uniform(0.5, 2.0)
        return CoherentState(alpha, cutoff), 1
