import os.path
import requests
from urllib.parse import urlencode
import zipfile

import click


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
@click.option('-bd', '--base_dir', default='..\\data\\raw')
@click.option('-dn', '--dataset_name', default='flowers')
@click.option('-od', '--out_dir', default='..\\data\\processed\\')
def processed_data(dataset_name, base_dir, out_dir):
    make_base_dir(base_dir)
    make_out_dir(out_dir, False)

    # this dictionary will be in the file datasets_info.json
    base_path = os.path.join(base_dir, (datasets_info[dataset_name]['folder'] + '.zip'))
    out_dir = os.path.join(out_dir, datasets_info[dataset_name]['folder'])
    make_out_dir(out_dir, True)

    if not(os.path.exists(base_path)):
        download_file_from_yandex_disk(
            datasets_info[dataset_name]['url'],
            base_path
        )

    with zipfile.ZipFile(base_path, 'r') as zf:
        zf.extractall(out_dir)


def download_file_from_yandex_disk(file_url, base_path):
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'  # Сюда вписываете вашу ссылку

    final_url = base_url + urlencode(dict(public_key=file_url))
    response = requests.get(final_url)
    download_url = response.json()['href']

    download_response = requests.get(download_url)
    with open(base_path, 'wb') as f:  # Здесь укажите нужный путь к файлу
        f.write(download_response.content)


def make_out_dir(out_dir, rm_flag=False):
    if os.path.exists(out_dir) and rm_flag:
        os.remove(out_dir)
        os.mkdir(out_dir)
    if not(os.path.exists(out_dir)):
        os.mkdir(out_dir)


def make_base_dir(base_dir):
    if not(os.path.exists(base_dir)):
        os.mkdir(base_dir)


if __name__ == '__main__':
    processed_data()
