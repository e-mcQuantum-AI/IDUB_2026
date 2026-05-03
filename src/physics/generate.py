"""Command-line dataset generator for quantum states.

This module provides a CLI pipeline for generating quantum state
datasets (Fock, cat, binomial, coherent) and saving their Wigner
representations as images.
"""

import argparse
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
    clean_path = config.output_dir / "clean"
    clean_path.mkdir(parents=True, exist_ok=True)

    return {
        "clean": clean_path,
    }


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

    print(f"Generowanie {config.n_samples} par stanów...")

    for i in range(config.n_samples):
        # --- 1. FOCK STATE ---
        n_photons: int = int(rng.integers(0, 8))
        state_fock = FockState(n=n_photons, cutoff=config.cutoff).ket()

        w_clean = wm.measure(state_fock)
        plt.imsave(paths["clean"] / f"fock_n{n_photons}_id{i}.png", w_clean, cmap=config.cmap)

        # --- 2. CAT STATE ---
        alpha_cat: complex = complex(rng.uniform(1.5, 3.5), rng.uniform(1.5, 3.5))
        state_cat = CatState(alpha=alpha_cat, cutoff=config.cutoff).ket()

        w_cat_clean = wm.measure(state_cat)
        plt.imsave(paths["clean"] / f"cat_a{alpha_cat:.2f}_id{i}.png", w_cat_clean, cmap=config.cmap)

        # --- 3. BINOMIAL STATE ---
        n: int = int(rng.integers(0, 8))
        p: float = float(rng.uniform(0.1, 0.9))
        state_binomial = BinomialState(N=n, p=p, cutoff=config.cutoff).ket()

        w_binomial_clean = wm.measure(state_binomial)
        plt.imsave(paths["clean"] / f"binomial_n{n}_p{p:.2f}_id{i}.png", w_binomial_clean, cmap=config.cmap)

        # --- 4. COHERENT STATE ---
        alpha_coherent: complex = complex(rng.uniform(1.5, 3.5), rng.uniform(1.5, 3.5))
        state_coherent = CoherentState(alpha=alpha_coherent, cutoff=config.cutoff).ket()

        w_coherent_clean = wm.measure(state_coherent)
        plt.imsave(paths["clean"] / f"coherent_a{alpha_coherent:.2f}_id{i}.png", w_coherent_clean, cmap=config.cmap)

    print(f"Dane zapisano w folderze {config.output_dir}")


def parse_args() -> GeneratorConfig:
    """Parse CLI arguments into a GeneratorConfig object.

    Returns:
        GeneratorConfig: Parsed configuration for dataset generation.
    """
    parser = argparse.ArgumentParser(description="Generowanie datasetu stanów kwantowych")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quantum_dataset"),
        help="Folder wyjściowy",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=32,
        help="Cutoff przestrzeni Focka",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=100,
        help="Rozdzielczość siatki Wignera",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        default=5.0,
        help="Zakres osi x",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Liczba próbek",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="RdBu_r",
        help="Mapa kolorów matplotlib",
    )

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
    config = parse_args()
    generate_dataset(config)

if __name__ == "__main__":
    main()
