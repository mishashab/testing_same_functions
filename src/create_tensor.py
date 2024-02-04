import os
import os.path
import click
import numpy as np
import tensorflow
import numpy
from PIL import Image


datasets_info = {
        'flowers': {
            'folder': 'fl',
            'url': 'https://disk.yandex.ru/d/qOyHCK5anbmD7w',
        },
        'cars': {
            'folder': "cr",
            'url': '',
        },
    }

@click.command()
@click.option('-bd', '--in_dir', default='..\\data\\processed')
@click.option('-dn', '--dataset_name', default='flowers')
@click.option('-od', '--out_dir', default='..\\data\\tensor\\')
def create_tensor(in_dir, out_dir, dataset_name):
    dir = os.path.join(in_dir, datasets_info[dataset_name]['folder'], 'train')
    print(dir)
    dataset = tensorflow.keras.preprocessing.image_dataset_from_directory(
        directory=dir,
        labels=None,
        label_mode='int',
        color_mode='rgb',
        image_size=(256, 256),
        shuffle=True,
        validation_split=0.2,
        subset='both',
        seed=42,
    )
    print('ok')
    print(type(dataset))


def make_dir(dir):
    if not (os.path.exists(dir)):
        os.mkdir(dir)


if __name__ == '__main__':
    create_tensor()
