"""
"""
import os
import sys

import requests

from bs4 import BeautifulSoup


def main():

    opj = os.path.join

    url = sys.argv[1]
    save_dir = sys.argv[2]

    os.makedirs(save_dir, exist_ok=True)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')

    links = []
    for img in soup.findAll('img'):
        if img not in links:
            links.append(img)

    f = open('tmp.txt', 'w')
    for img in links:
        f.write('https:' + img.get('src') + '\n')
    f.close()
    os.system('wget --directory='+save_dir + ' --input-file=tmp.txt')


if __name__ == '__main__':
    main()
