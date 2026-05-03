"""
CLI entrypoint for quantum dataset generation.

This script provides a convenient wrapper for running the dataset
generation pipeline from the project root.

Example:
    python generate.py --help
"""

from src.physics.generate import main


if __name__ == "__main__":
    main()
