import os
from PIL import Image


def divide_on_squares(img_name, x, output_directory='divided'):
    name = os.path.splitext(img_name)[0]
    output_directory = f'{output_directory}/{name}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    img = Image.open(f'quantum_dataset/clean/{img_name}')
    width, height = img.size

    counter = 0

    for i in range(0, height, x):
        for j in range(0, width, x):
            if i + x <= height and j + x <= width:
                box = (j, i, j + x, i + x)
                fragment = img.crop(box)

                file_name = f'{name}_{i // x}_{j // x}.png'
                fragment.save(os.path.join(output_directory, file_name))
                counter += 1

    print(f'Saved in {output_directory}')


if __name__ == '__main__':
    divide_on_squares('fock_n0_id0.png', 50)
