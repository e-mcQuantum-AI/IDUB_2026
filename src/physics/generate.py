"""Command-line dataset generator for quantum states.

This module provides a CLI pipeline for generating quantum state
datasets (Fock, cat, binomial, coherent) and saving their Wigner
representations as images.
"""

import argparse
import csv
import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from . import WignerMeasurement, FockState, CatState, BinomialState, CoherentState, GeneratorConfig


def setup_folders(config: GeneratorConfig) -> dict[str, Path]:
    """Create output directory structure for dataset generation.

    Args:
        config (GeneratorConfig): Global dataset configuration.

    Returns:
        dict[str, Path]: Dictionary containing paths for dataset splits.
    """
    paths = {
        "clean": config.output_dir / "clean",
        "loss_0.1": config.output_dir / "loss_0.1",
        "loss_0.3": config.output_dir / "loss_0.3",
        "dephasing_0.2": config.output_dir / "dephasing_0.2",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def apply_noise(state, noise_type: str, gamma: float, cutoff: int):
    """Applies a quantum noise channel using the Lindblad master equation.

    Args:
        state: Input quantum state (ket or density matrix).
        noise_type (str): Type of noise ('loss' or 'dephasing').
        gamma (float): Channel strength (decay rate * time).
        cutoff (int): Hilbert space dimension.

    Returns:
        Density matrix (mixed state) after the noise channel.
    """

    if not isinstance(state, qt.Qobj):
        state = qt.Qobj(state)

    if state.isket:
        rho0 = state * state.dag()
    else:
        rho0 = state

    a = qt.destroy(cutoff)
    n = qt.num(cutoff)

    if noise_type == "loss":
        c_ops = [np.sqrt(gamma) * a]
    elif noise_type == "dephasing":
        c_ops = [np.sqrt(gamma) * n]
    else:
        return rho0

    result = qt.mesolve(qt.qzero(cutoff), rho0, [0, 1.0], c_ops=c_ops)
    return result.states[-1]


def generate_dataset(config: GeneratorConfig) -> None:
    """Generate quantum dataset and save Wigner images.

    This function generates multiple types of quantum states, computes
    their Wigner functions, and saves them as images.

    Args:
        config (GeneratorConfig): Dataset generation configuration.
    """
    paths = setup_folders(config)

    wm = WignerMeasurement(
        x_max=config.x_range,
        resolution=config.resolution,
    )

    rng = np.random.default_rng()
    csv_path = config.output_dir / "metadata.csv"

    print(f"Generowanie {config.n_samples} par stanów wraz z wariantami zaszumionymi...")

    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Nagłówek metadanych
            writer.writerow(["filename", "state_type", "noise_type", "noise_param", "n_photons", "alpha", "p"])

            for i in range(config.n_samples):
                # --- 1. FOCK STATE (Czysty + Straty) ---
                n_photons: int = int(rng.integers(0, 8))
                state_fock = FockState(n=n_photons, cutoff=config.cutoff).ket()

                # Clean
                fname = f"fock_n{n_photons}_id{i}.png"
                plt.imsave(paths["clean"] / fname, wm.measure(state_fock), cmap=config.cmap)
                writer.writerow([fname, "fock", "none", "", n_photons, "", ""])

                # Loss 0.1
                rho_fock_loss1 = apply_noise(state_fock, "loss", 0.1, config.cutoff)
                fname = f"fock_n{n_photons}_loss0.1_id{i}.png"
                plt.imsave(paths["loss_0.1"] / fname, wm.measure(rho_fock_loss1), cmap=config.cmap)
                writer.writerow([fname, "fock", "loss", 0.1, n_photons, "", ""])

                # Loss 0.3
                rho_fock_loss3 = apply_noise(state_fock, "loss", 0.3, config.cutoff)
                fname = f"fock_n{n_photons}_loss0.3_id{i}.png"
                plt.imsave(paths["loss_0.3"] / fname, wm.measure(rho_fock_loss3), cmap=config.cmap)
                writer.writerow([fname, "fock", "loss", 0.3, n_photons, "", ""])

                # --- 2. CAT STATE (Czysty + Straty + Defazowanie) ---
                alpha_cat: complex = complex(rng.uniform(1.5, 3.5), rng.uniform(1.5, 3.5))
                state_cat = CatState(alpha=alpha_cat, cutoff=config.cutoff).ket()
                alpha_str = f"{alpha_cat:.2f}"

                # Clean
                fname = f"cat_a{alpha_str}_id{i}.png"
                plt.imsave(paths["clean"] / fname, wm.measure(state_cat), cmap=config.cmap)
                writer.writerow([fname, "cat", "none", "", "", alpha_cat, ""])

                # Zdekohowany Kot (Loss 0.1)
                rho_cat_loss = apply_noise(state_cat, "loss", 0.1, config.cutoff)
                fname = f"cat_a{alpha_str}_loss0.1_id{i}.png"
                plt.imsave(paths["loss_0.1"] / fname, wm.measure(rho_cat_loss), cmap=config.cmap)
                writer.writerow([fname, "cat", "loss", 0.1, "", alpha_cat, ""])

                # Zdekohowany Kot (Dephasing 0.2)
                rho_cat_deph = apply_noise(state_cat, "dephasing", 0.2, config.cutoff)
                fname = f"cat_a{alpha_str}_dephasing0.2_id{i}.png"
                plt.imsave(paths["dephasing_0.2"] / fname, wm.measure(rho_cat_deph), cmap=config.cmap)
                writer.writerow([fname, "cat", "dephasing", 0.2, "", alpha_cat, ""])

                # --- 3. BINOMIAL STATE (Czysty) ---
                n: int = int(rng.integers(0, 8))
                p: float = float(rng.uniform(0.1, 0.9))
                state_binomial = BinomialState(N=n, p=p, cutoff=config.cutoff).ket()

                # Clean
                fname = f"binomial_n{n}_p{p:.2f}_id{i}.png"
                plt.imsave(paths["clean"] / fname, wm.measure(state_binomial), cmap=config.cmap)
                writer.writerow([fname, "binomial", "none", "", n, "", p])

                # --- 4. COHERENT STATE (Czysty + Defazowanie) ---
                alpha_coherent: complex = complex(rng.uniform(1.5, 3.5), rng.uniform(1.5, 3.5))
                state_coherent = CoherentState(alpha=alpha_coherent, cutoff=config.cutoff).ket()
                alpha_coh_str = f"{alpha_coherent:.2f}"

                # Clean
                fname = f"coherent_a{alpha_coh_str}_id{i}.png"
                plt.imsave(paths["clean"] / fname, wm.measure(state_coherent), cmap=config.cmap)
                writer.writerow([fname, "coherent", "none", "", "", alpha_coherent, ""])

                # Dephasing 0.2
                rho_coh_deph = apply_noise(state_coherent, "dephasing", 0.2, config.cutoff)
                fname = f"coherent_a{alpha_coh_str}_dephasing0.2_id{i}.png"
                plt.imsave(paths["dephasing_0.2"] / fname, wm.measure(rho_coh_deph), cmap=config.cmap)
                writer.writerow([fname, "coherent", "dephasing", 0.2, "", alpha_coherent, ""])

    except ValueError as e:
        raise ValueError(f"Błąd wartości podczas generowania datasetu: {e}")
    except OSError as e:
        raise OSError(f"Błąd systemu plików podczas zapisu datasetu: {e}")
    except Exception as e:
        raise RuntimeError(f"Nieoczekiwany błąd podczas generowania datasetu: {e}")

    print(f"Dane oraz plik metadata.csv zapisano w folderze {config.output_dir}")


def parse_args() -> GeneratorConfig:
    """Parse CLI arguments into a GeneratorConfig object.

    Returns:
        GeneratorConfig: Parsed configuration for dataset generation.
    """
    parser = argparse.ArgumentParser(description="Generowanie datasetu stanów kwantowych")
    parser.add_argument("--output-dir", type=Path, default=Path("quantum_dataset"), help="Folder wyjściowy")
    parser.add_argument("--cutoff", type=int, default=32, help="Cutoff przestrzeni Focka")
    parser.add_argument("--resolution", type=int, default=100, help="Rozdzielczość siatki Wignera")
    parser.add_argument("--x-range", type=float, default=5.0, help="Zakres osi x")
    parser.add_argument("--n-samples", type=int, default=10, help="Liczba próbek do wygenerowania na typ")
    parser.add_argument("--cmap", type=str, default="RdBu_r", help="Mapa kolorów matplotlib")
    args = parser.parse_args()

    return GeneratorConfig(
        output_dir=args.output_dir,
        cutoff=args.cutoff,
        resolution=args.resolution,
        x_range=args.x_range,
        n_samples=args.n_samples,
        cmap=args.cmap,
    )


def main():
    """CLI entry point for dataset generation."""
    try:
        config = parse_args()
        generate_dataset(config)
    except FileNotFoundError as e:
        print(f"[FileNotFoundError] {e}")
    except ValueError as e:
        print(f"[ValueError] {e}")
    except OSError as e:
        print(f"[OSError] {e}")
    except RuntimeError as e:
        print(f"[RuntimeError] {e}")


if __name__ == "__main__":
    main()
