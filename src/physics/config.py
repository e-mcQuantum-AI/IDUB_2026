"""Configuration for quantum dataset generation.

This module defines parameters used for generating quantum states
and their representations (e.g., Wigner functions).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GeneratorConfig:
    """Configuration parameters for quantum data generation.

    This dataclass defines global settings used in dataset generation,
    including Hilbert space truncation, sampling resolution, and output
    visualization parameters.

    Attributes:
        output_dir (Path): Directory where generated data is stored.
        cutoff (int): Hilbert space dimension truncation.
        resolution (int): Grid resolution for phase-space sampling.
        x_range (float): Range of phase-space coordinates.
        n_samples (int): Number of samples to generate.
        cmap (str): Colormap used for visualization.
    """

    output_dir: Path = Path("quantum_dataset")
    cutoff: int = 32
    resolution: int = 100
    x_range: float = 5.0
    n_samples: int = 10
    cmap: str = "RdBu_r"
