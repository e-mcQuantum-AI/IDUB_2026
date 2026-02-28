import os
import matplotlib.pyplot as plt
import numpy as np
from qutip import qeye, wigner, destroy, fock_dm

class State_Database_Generator:
    def __init__(self, cutoff: int, N: int, root_dir: str = "data"):
        self.cutoff = cutoff
        self.N = N
        self.root_dir = root_dir
        self.rng = np.random.default_rng()
        self.identity = qeye(self.cutoff) / self.cutoff
        self.a = destroy(self.cutoff)
        
        os.makedirs(self.root_dir, exist_ok=True)

    def apply_noise(self, rho, noise_type, strength):
        if noise_type == "depolarization":
            # Obecne mieszanie macierzy gęstości z identycznością
            return (1 - strength) * rho + (strength * self.identity / self.cutoff)
            
        elif noise_type == "photon_loss":
            # Uproszczony model utraty fotonów (kanał tłumiący)
            # Dla potrzeb ML: przesuwamy stan w stronę próżni
            vacuum = fock_dm(self.cutoff, 0)
            return (1 - strength) * rho + (strength * vacuum)
            
        elif noise_type == "gaussian":
            # Dodanie losowego przesunięcia (displacement) w przestrzeni fazowej
            alpha = (self.rng.standard_normal() + 1j * self.rng.standard_normal()) * strength
            D = (alpha * self.a.dag() - np.conj(alpha) * self.a).expm()
            return D * rho * D.dag()
            
        return rho
    
    def generate(self, state_class, noise_type, **state_kwargs):
        save_path = os.path.join(self.root_dir, state_class.__name__)
        os.makedirs(save_path, exist_ok=True)

        # identyfikacja stanu który chcemy generować
        quantum_state = state_class(cutoff=self.cutoff, **state_kwargs)
        rho = quantum_state.density_matrix()

        print(f"Generujemy {self.N} plików dla {state_class.__name__}...")

        # pętla generująca N stanów
        for i in range(self.N):
            p = self.rng.uniform(0, 0.3)
            rho_noise = self.apply_noise(rho, noise_type, p)

            xvec = np.linspace(-5, 5, 200)
            W = wigner(rho_noise, xvec, xvec) # wigner, bardzo ważny

            fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1]) # Obraz zajmuje całe pole, bez marginesów
            
            # wizualizacja funkcji Wignera
            ax.imshow(W, cmap='RdBu', extent=[-5, 5, -5, 5], origin="lower")
            
            # Usuń śmieci 
            ax.axis('off')
            
            # zapis
            file_name = f"{state_class.__name__}_ID{i:04d}_p{p:.4f}.png"
            plt.savefig(os.path.join(save_path, file_name), format='png')