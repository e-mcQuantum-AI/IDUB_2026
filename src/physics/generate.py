import os
import numpy as np
import matplotlib.pyplot as plt
from physics import CatState, FockState, BinomialState, WignerMeasurement, CoherentState


def setup_folders():
    base_path = "quantum_dataset"
    folders = ["clean"]
    paths = {}
    for f in folders:
        path = os.path.join(base_path, f)
        if not os.path.exists(path):
            os.makedirs(path)
        paths[f] = path
    return paths

def generate_dataset(n_samples=10, resolution=100):
    paths = setup_folders()
    N = 32
    wigner = WignerMeasurement(x_max=5, resolution=resolution)
    rng = np.random.default_rng()
    
    print(f"Generowanie {n_samples} par stanów...")

    for i in range(n_samples):
        # --- 1. STAN FOCKA ---
        n_photons: int = int(rng.integers(0, 8))
        state_fock = FockState(n=n_photons, cutoff=N).ket()

        # Czysty
        w_clean = wigner.measure(state_fock)
        plt.imsave(os.path.join(paths["clean"], f"fock_n{n_photons}_id{i}.png"), w_clean, cmap='RdBu_r')

        # --- 2. STAN KOTA ---
        alpha: complex = complex(rng.uniform(1.5, 3.5), rng.uniform(1.5, 3.5))
        state_cat = CatState(alpha=alpha, cutoff=N).ket()

        # Czysty
        w_cat_clean = wigner.measure(state_cat)
        plt.imsave(os.path.join(paths["clean"], f"cat_a{alpha:.2f}_id{i}.png"), w_cat_clean, cmap='RdBu_r')     

        # --- 3. STAN BINOMIALNY ---
        n: int = int(rng.integers(0, 8))
        p: float = float(rng.uniform(0.1, 0.9))
        state_binomial = BinomialState(N=n, p=p, cutoff=N).ket()

        # Czysty
        w_binomial_clean = wigner.measure(state_binomial)
        plt.imsave(os.path.join(paths["clean"], f"binomial_n{n}_p{p:.2f}_id{i}.png"), w_binomial_clean, cmap='RdBu_r')

        # --- 4. STAN KOHERENTNY ---
        alpha: complex = complex(rng.uniform(1.5, 3.5), rng.uniform(1.5, 3.5))
        state_coherent = CoherentState(alpha=alpha, cutoff=N).ket()

        # Czysty
        w_coherent_clean = wigner.measure(state_coherent)
        plt.imsave(os.path.join(paths["clean"], f"coherent_a{alpha:.2f}_id{i}.png"), w_coherent_clean, cmap='RdBu_r')

    print("Dane zapisano w folderze quantum_dataset/")

generate_dataset(n_samples=5)
