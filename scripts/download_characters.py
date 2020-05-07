"""
"""
import os
import urllib
import urllib.request

import requests

from bs4 import BeautifulSoup


def get_character_links():
    """ Not needed anymore, but good to save """
    page = requests.get(
            'https://bulbapedia.bulbagarden.net/wiki/List_of_game_characters')
    soup = BeautifulSoup(page.text, 'html.parser')
    links = soup.find_all('a')
    f = open(os.path.join('data', 'character_links.txt'), 'w')
    for link in links:
        url = link.get('href')
        if url is None:
            continue
        if '.png' in url:
            continue
        f.write(url+'\n')
    f.close()


def main():

    opj = os.path.join
    base_url = 'https://bulbapedia.bulbagarden.net'
    save_dir = os.path.join('data', 'characters')

    character_links = []
    with open(os.path.join('data', 'character_links.txt'), 'r') as f:
        for line in f:
            line = line.rstrip()
            line = base_url + line
            character_links.append(line)

    for character_link in character_links:
        character_name = urllib.parse.unquote(
                character_link.split('wiki/')[-1].split('_(')[0])
        print('Getting urls for', character_name)

        character_dir = opj(save_dir, character_name)
        character_file = opj(character_dir, 'images.txt')
        os.makedirs(character_dir, exist_ok=True)
        page = requests.get(character_link)
        soup = BeautifulSoup(page.text, 'html.parser')

        f = open(character_file, 'w')
        for img in soup.findAll('img'):
            f.write('https:' + img.get('src') + '\n')
        f.close()
        os.system('wget --directory='+character_dir + ' --input-file='+character_file)


if __name__ == '__main__':
    main()
