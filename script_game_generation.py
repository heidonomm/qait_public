import tempfile
import os
from os.path import join as pjoin
from distutils.dir_util import copy_tree
import glob

from textworld.gym import register_games
import numpy as np
import gym
import textworld

import game_generator

# request_infos = textworld.EnvInfos(description=True,
#                                    inventory=True,
#                                    verbs=True,
#                                    location_names=True,
#                                    location_nouns=True,
#                                    location_adjs=True,
#                                    object_names=True,
#                                    object_nouns=True,
#                                    object_adjs=True,
#                                    facts=True,
#                                    last_action=True,
#                                    game=True,
#                                    admissible_commands=True,
#                                    extras=["object_locations", "object_attributes", "uuid"])

# games_dir = tempfile.TemporaryDirectory(prefix="tw_games")
# print(games_dir.name)

with tempfile.TemporaryDirectory(prefix="tw_games") as games_dir:
    assert os.path.exists(
        "./textworld_data"), "Oh no! textworld_data folder is not there..."
    # os.mkdir(games_dir)
    os.mkdir(pjoin(games_dir, "textworld_data"))
    copy_tree("textworld_data", games_dir + "textworld_data")
    print(os.listdir(games_dir))
    print(os.listdir(games_dir + "/textworld_data"))
    print(games_dir)

    game_generator_queue = game_generator.game_generator(
        path=games_dir, random_map=False, question_type="location", train_data_size=3
    )
    files_in_dir = os.listdir(games_dir)
    print(files_in_dir)

# tmp_dir = pjoin(tmp_dir, "")
# print(type(tmp_dir))
# print(f"tmp_dir is a string -> {tmp_dir}")

# with open(f"{tmp_dir}file.txt", "w+", encoding="utf-8") as f:
#     f.write("kalkun")

# # get files from the temp directory
# files_in_dir = os.listdir(tmp_dir)
# print(files_in_dir)


# games_dir = tempfile.TemporaryDirectory(prefix="tw_games", dir="./")
# print(games_dir.name)
# print(os.listdir("./"))
# print(os.listdir("./kgdqn"))

# games_dir = pjoin(games_dir.name, "")  # So path ends with '/'.

# # copy grammar files into tmp folder so that it works smoothly
# assert os.path.exists(
#     "./textworld_data"), "Oh no! textworld_data folder is not there..."
# os.mkdir(games_dir)
# os.mkdir(pjoin(games_dir, "textworld_data"))
# copy_tree("textworld_data", games_dir + "textworld_data")

# # generate the training set
# all_training_games = game_generator.game_generator(
#     path=games_dir, random_map=False, question_type="location", train_data_size=5)
# all_training_games.sort()
# all_env_ids = [register_game(
#     gamefile, request_infos=request_infos) for gamefile in all_training_games]
# env_ids = np.random.choice(all_env_ids, 1).tolist()
# env_id = make_batch2(env_ids, parallel=True)
# print(os.listdir("./"))
# print(os.listdir("./kgdqn"))


# games = glob.glob(games_dir + '*.ulx')[:2]
# print(games)

# env = gym.make(env_id)
# env.seed(10)
# print(env.game)
# self._commands = iter(env.game.quests[0].commands)


# print(os.listdir(games_dir))
# print(os.listdir(games_dir + "/textworld_data"))
