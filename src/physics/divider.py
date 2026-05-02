import os
from PIL import Image


def divide_on_squares(img_name: str, x: int, output_directory: str= 'divided'):
    name: str = os.path.splitext(img_name)[0]
    output_dir: str = f'{output_directory}/{name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = Image.open(f'quantum_dataset/clean/{img_name}')
    width, height = img.size

    counter: int = 0

    for i in range(0, height, x):
        for j in range(0, width, x):
            if i + x <= height and j + x <= width:
                box = (j, i, j + x, i + x)
                fragment = img.crop(box)

                file_name: str = f'{name}_{i // x}_{j // x}.png'
                fragment.save(os.path.join(output_dir, file_name))
                counter += 1

    print(f'Saved in {output_dir}')


if __name__ == '__main__':
    divide_on_squares('fock_n0_id0.png', 50)
