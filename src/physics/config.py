from dataclasses import dataclass
from pathlib import Path

@dataclass
class GeneratorConfig:
    output_dir: Path = Path("quantum_dataset")
    cutoff: int = 32
    resolution: int = 100
    x_range: float = 5.0
    n_samples: int = 10
    cmap: str = "RdBu_r"
