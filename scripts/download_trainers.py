"""
"""
import os
import urllib
import urllib.request

import requests

from bs4 import BeautifulSoup


def get_trainer_class_links():
    """ Not needed anymore, but good to save """
    base_url = 'https://bulbapedia.bulbagarden.net/'
    tc_url = base_url + 'wiki/Pok%C3%A9mon_Trainer#List_of_Trainer_classes'
    page = requests.get(tc_url)
    soup = BeautifulSoup(page.text, 'html.parser')
    trainer_class_links = []
    for link in soup.find_all('a'):
        link_name = str(link.get('href'))
        if '_(Trainer_class)' in link_name:
            trainer_class_links.append(link_name)
    with open('data/trainer_class_links.txt', 'w') as f:
        for link in trainer_class_links:
            f.write(link+'\n')


def main():

    opj = os.path.join

    base_url = 'https://bulbapedia.bulbagarden.net/'
    save_dir = os.path.join('data', 'trainer_class')

    trainer_class_links = []
    with open(os.path.join('data', 'trainer_class_links.txt'), 'r') as f:
        for line in f:
            line = line.rstrip()
            line = base_url + line
            trainer_class_links.append(line)

    for trainer_link in trainer_class_links:
        trainer_name = urllib.parse.unquote(
                trainer_link.split('wiki/')[-1].split('_(')[0])
        print('Getting urls for', trainer_name)

        trainer_dir = opj(save_dir, trainer_name)
        trainer_file = opj(trainer_dir, 'images.txt')
        os.makedirs(trainer_dir, exist_ok=True)
        page = requests.get(trainer_link)
        soup = BeautifulSoup(page.text, 'html.parser')

        f = open(trainer_file, 'w')
        for img in soup.findAll('img'):
            f.write('https:' + img.get('src') + '\n')
        f.close()
        os.system('wget --directory='+trainer_dir + ' --input-file='+trainer_file)


if __name__ == '__main__':
    main()
