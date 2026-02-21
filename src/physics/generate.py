import os
import matplotlib.pyplot as plt
import numpy as np
from qutip import qeye, wigner 

class State_Database_Generator:
    def __init__(self, cutoff: int, N: int, root_dir: str = "data"):
        self.cutoff = cutoff
        self.N = N
        self.root_dir = root_dir
        self.rng = np.random.default_rng()
        self.identity = qeye(self.cutoff) / self.cutoff

    def generate(self, state_class, folder_name, **state_kwargs):
        save_path = os.path.join(self.root_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)

        quantum_state = state_class(cutoff=self.cutoff, **state_kwargs)
        rho = quantum_state.density_matrix()

        print(f"Generujemy {self.N} plików...")

        for i in range(self.N):
            p = self.rng.random()
            rho_noise = p * rho + (1 - p) * self.identity

            xvec = np.linspace(-5, 5, 200)
            W = wigner(rho_noise, xvec, xvec)

            fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1]) # Obraz zajmuje całe pole, bez marginesów
            
            # funkcja Wignera
            ax.imshow(W, cmap='RdBu', extent=[-5, 5, -5, 5])
            
            # Usuń śmieci 
            ax.axis('off')
            
            file_name = f"{folder_name}_ID{i:04d}_p{p:.4f}.png"
            plt.savefig(os.path.join(save_path, file_name), format='png')