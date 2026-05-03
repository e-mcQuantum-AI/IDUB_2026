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
        OSError: For filesystem or image I/O related issues.
    """
    if x <= 0:
        raise ValueError("x must be positive.")

    input_path = os.path.join("quantum_dataset", "clean", img_name)

    name = os.path.splitext(img_name)[0]
    output_dir = os.path.join(output_directory, name)
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(input_path)
    width, height = img.size

    counter = 0

    for i in range(0, height, x):
        for j in range(0, width, x):
            if i + x <= height and j + x <= width:
                box = (j, i, j + x, i + x)
                fragment = img.crop(box)

                file_name = f"{name}_{i // x}_{j // x}.png"
                fragment.save(os.path.join(output_dir, file_name))
                counter += 1

    print(f"Saved {counter} tiles in {output_dir}")


if __name__ == "__main__":
    try:
        divide_on_squares("fock_n0_id0.png", 50)
    except FileNotFoundError as e:
        print(f"[File error] {e}")
    except ValueError as e:
        print(f"[Value error] {e}")
    except OSError as e:
        print(f"[OS error] {e}")
