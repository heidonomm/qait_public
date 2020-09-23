from gdqn import KGDQNTrainer
from joblib import Parallel, delayed

import glob
import tempfile
import os
from os.path import join as pjoin
from distutils.dir_util import copy_tree
import game_generator
import textworld
from textworld.gym import register_games
import numpy as np


request_infos = textworld.EnvInfos(description=True,
                                   inventory=True,
                                   verbs=True,
                                   facts=True,
                                   last_action=True,
                                   game=True,
                                   admissible_commands=True,
                                   intermediate_reward=True,
                                   extras=["object_locations", "object_attributes", "uuid"])


def parallelize(params):
    print(params)
    generated_path = generate_games()
    # print(f"generated game with batch2 is {generated_path}")
    # trainer.generate_text_files(generated_path)
    # games = glob.glob(generated_path + "*.ulx")
    trainer = KGDQNTrainer(params, generated_path)
    trainer.train()


def generate_games():
    games_dir = tempfile.TemporaryDirectory(
        prefix="tw_games", dir="./kgdqn")
    games_dir = pjoin(games_dir.name, "")  # So path ends with '/'.
    # copy grammar files into tmp folder so that it works smoothly
    assert os.path.exists(
        "textworld_data"), "Oh no! textworld_data folder is not there..."
    os.mkdir(games_dir)
    os.mkdir(pjoin(games_dir, "textworld_data"))
    copy_tree("textworld_data", games_dir + "textworld_data")

    # generate the training set of games
    all_games_path = game_generator.game_generator(
        path=games_dir, random_map=False, question_type="location", train_data_size=4)
    all_games_path.sort()
    print(f"all games are in following paths: {all_games_path}")
    print(f"os listed files in generated path {os.listdir(games_dir)}")

    # all_env_ids = register_games(
    #     gamefiles=all_games_path, request_infos=request_infos)

    # print(f"all_env_ids are {all_env_ids}")

    # chosen_ids = np.random.choice(all_env_ids, 4).tolist()
    # print(f"chosen_ids are {chosen_ids}")
    # current_game_id = make_batch2(chosen_ids, parallel=True)
    # print(f"current_game_id is {current_game_id}")

    # print(f"temporary games directory: {games_dir}")
    return all_games_path


if __name__ == "__main__":
    # Example for random grid search on the parameter space
    """
    param_grid = {
        'replay_buffer_type': ['priority', 'standard'],
        'replay_buffer_size': [10000, 50000],
        'num_frames': [100000, 500000],
        'batch_size': [64],
        'lr': [0.01, 0.001],
        'gamma': [0.5, 0.2, 0.5],
        'rho': [0.25],
        'scheduler_type': ['exponential', 'linear'],
        'e_decay': [500, 10000, 20000, 50000],
        'e_final': [0.01, 0.1, 0.2],
        'hidden_dims': [[64, 32], [128, 64], [256, 128]],
        'update_frequency': [1, 4, 10]
    }
    """

    # grid_search = RandomGridSearch(param_grid, 0.2, 21)

    # insert one ulx file generated by tw-make here
    game = "*.ulx"

    # all_params = grid_search.get_configs()
    # parallelize(game, all_params[0])

    # Uncomment and define cuda visible device to parallelize across multiple processes

    # Parallel(n_jobs=2, prefer='processes', backend='multiprocessing')(
    #    delayed(parallelize)(game, params) for params in all_params)
    params = {
        'replay_buffer_type': 'priority',
        'replay_buffer_size': 100000,
        'num_episodes': 5000,
        'num_frames': 5000,
        'batch_size': 32,
        'lr': 0.001,
        'gamma': 0.5,
        'rho': 0.25,
        'scheduler_type': 'exponential',
        'e_decay': 10000,
        'e_final': 0.2,
        'hidden_dims': 0,
        'update_frequency': 5,
        'padding_idx': 0,
        'embedding_size': 50,
        'dropout_ratio': 0.2,
        'hidden_size': 100,
        'gat_emb_size': 50,
        'drqa_emb_size': 384,
        'gat_emb_init_file': '',
        'act_emb_init_file': '',
        'preload_weights': False,
        'preload_file': '',
        'pruned': False,
        'max_actions': 40,
        'init_graph_embeds': True,
        'qa_init': False,
        'vocab_size': 1000,
        'cuda_device': 1,
        'gameid': 0,
        'doc_hidden_size': 64,
        'doc_layers': 3,
        'doc_dropout_rnn': 0.2,
        'doc_dropout_rnn_output': True,
        'doc_concat_rnn_layers': True,
        'doc_rnn_padding': True
    }

    drqa_params = {
        'doc_hidden_size': 64,
        'doc_layers': 3,
        'doc_dropout_rnn': 0.2,
        'doc_dropout_rnn_output': True,
        'doc_concat_rnn_layers': True,
        'doc_rnn_padding': True

    }

    qait_params = {
        "aggregation_layers": 3,
        "aggregation_conv_num": 2,
        "block_hidden_dim": 64,
        "n_heads": 1,
        "attention_dropout": 0.,
        "block_dropout": 0.,
        "noisy_net": False

    }

    params.update(drqa_params)
    params.update(qait_params)
    parallelize(params)
