import os
import numpy as np
import matplotlib.pyplot as plt
from qutip import fock, coherent, wigner, ket2dm

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
    xvec = np.linspace(-5, 5, resolution)
    
    print(f"Generowanie {n_samples} par stanów...")

    for i in range(n_samples):
        # --- 1. STAN FOCKA ---
        n_photons = np.random.randint(0, 8)
        state_fock = fock(N, n_photons)
        
        # Czysty
        w_clean = wigner(state_fock, xvec, xvec)
        plt.imsave(os.path.join(paths["clean"], f"fock_n{n_photons}_id{i}.png"), w_clean, cmap='RdBu_r')

        # --- 2. STAN KOTA ---
        alpha = np.random.uniform(1.5, 3.5)
        state_cat = (coherent(N, alpha) + coherent(N, -alpha)).unit()
        
        # Czysty
        w_cat_clean = wigner(state_cat, xvec, xvec)
        plt.imsave(os.path.join(paths["clean"], f"cat_a{alpha:.2f}_id{i}.png"), w_cat_clean, cmap='RdBu_r')     

        # --- 3. STAN BINOMIALNY ---
        n = np.random.randint(0, 8)
        p = np.random.uniform(0.1, 0.9)
        state_binomial = (fock(N, n) + p * fock(N, n+1)).unit()

        # Czysty
        w_binomial_clean = wigner(state_binomial, xvec, xvec)
        plt.imsave(os.path.join(paths["clean"], f"binomial_n{n}_p{p:.2f}_id{i}.png"), w_binomial_clean, cmap='RdBu_r')

        # --- 4. STAN KOHERENTNY ---
        alpha = np.random.uniform(1.5, 3.5)
        state_coherent = coherent(N, alpha)

        # Czysty
        w_coherent_clean = wigner(state_coherent, xvec, xvec)
        plt.imsave(os.path.join(paths["clean"], f"coherent_a{alpha:.2f}_id{i}.png"), w_coherent_clean, cmap='RdBu_r')

    print("Dane zapisano w folderze quantum_dataset/")

generate_dataset(n_samples=5)
