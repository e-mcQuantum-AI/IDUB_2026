"""Image tiling utility.

This module provides functionality for dividing quantum dataset images
into smaller square patches for further processing or training.
"""

import os
from PIL import Image


def divide_on_squares(img_name: str, x: int, output_directory: str = "divided") -> None:
    """Divide an image into square patches and save them to disk.

    The input image is split into non-overlapping square tiles of size
    x × x. Each tile is saved as a separate PNG file.

    Args:
        img_name (str): Name of the input image file (located in
            'quantum_dataset/clean/').
        x (int): Side length of each square patch in pixels.
        output_directory (str): Root directory where output folders
            will be created.

    Raises:
        FileNotFoundError: If the input image does not exist.
        ValueError: If x <= 0.
    """
    if x <= 0:
        raise ValueError("x must be positive.")

    input_path = os.path.join("quantum_dataset", "clean", img_name)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Image not found: {input_path}")

    name: str = os.path.splitext(img_name)[0]
    output_dir: str = f"{output_directory}/{name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = Image.open(input_path)
    width, height = img.size

    counter: int = 0

    for i in range(0, height, x):
        for j in range(0, width, x):
            if i + x <= height and j + x <= width:
                box = (j, i, j + x, i + x)
                fragment = img.crop(box)

                file_name: str = f"{name}_{i // x}_{j // x}.png"
                fragment.save(os.path.join(output_dir, file_name))
                counter += 1

    print(f"Saved {counter} tiles in {output_dir}")


if __name__ == "__main__":
    divide_on_squares("fock_n0_id0.png", 50)
