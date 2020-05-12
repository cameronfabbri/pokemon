"""

"""
# Copyright (c) Cameron Fabbri
# All rights reserved
import os


class Pokemon:
    """ Class that holds information for a single pokemon """

    def __init__(
            self,
            pid,
            paths):

        self.pid = pid

        self.paths = paths


class PokemonData:
    """

    `id_game_direction_shiny_icon_mega_num` where each is
    `int_str_str_bool_bool_bool_int`.

    """

    def __init__(
            self,
            data_dir):

        self.data_dir = data_dir

        self.filenames = os.listdir(data_dir)
        self.all_paths = [os.path.join(self.data_dir, x) for x in self.filenames]

        # Get all the names of the games in our dataset
        self.games = []

        # Get all of the pokemon ids we have in our dataset
        self.pokemon_ids = []

        # List of pokemon classes
        self.pokemon = {}

        for path in self.all_paths:

            path_split = os.path.basename(path).split('_')

            pid = int(path_split[0])
            game = path_split[1]
            #direction = path_split[2]
            #shiny = path_split[3]
            #icon = path_split[4]
            #mega = path_split[5]
            #num = path_split[6:]

            if pid not in self.pokemon_ids:
                self.pokemon_ids.append(pid)

            if game not in self.games:
                self.games.append(game)

            if pid not in self.pokemon.keys():
                self.pokemon[pid] = [path]
            else:
                self.pokemon[pid].append(path)


        self.games = sorted(self.games)
        self.pokemon_ids = sorted(self.pokemon_ids)

        self.generations = {}
        self.generations[1] = ['rb', 'green', 'yellow']
        self.generations[2] = ['gold', 'silver', 'crystal']
        self.generations[4] = ['hgss']
        self.generations[5] = ['bw']
        self.generations[9] = ['dreamworld']

    def get_paths_from_gen(self, gen):

        paths = []
        for path in self.all_paths:
            path_split = os.path.basename(path).split('_')
            game = path_split[1]
            if game in self.generations[gen]:
                paths.append(path)
        return paths

    def get_game(self, game):
        """ Returns paths to images from the given game """
        return None

    def get_pokemon(
            self,
            pokemon_id,
            game=None):
        """ """

        if game is not None:
            # Get from specific game
            pass

        return None

    def get_generation(
            self,
            gen_id):
        """ Returns paths to images from the given generation """
        return None


