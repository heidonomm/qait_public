from gdqn import KGDQNTrainer
from utils.grid_search import RandomGridSearch
from joblib import Parallel, delayed


def parallelize(game, params):
    print(params)
    trainer = KGDQNTrainer(game, params)
    trainer.train()


if __name__ == "__main__":
    #Example for random grid search on the parameter space
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

    #grid_search = RandomGridSearch(param_grid, 0.2, 21)

    #insert one ulx file generated by tw-make here
    game = "*.ulx"

    #all_params = grid_search.get_configs()
    # parallelize(game, all_params[0])

    #Uncomment and define cuda visible device to parallelize across multiple processes

    #Parallel(n_jobs=2, prefer='processes', backend='multiprocessing')(
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
        'qa_init': True,
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
    params.update(drqa_params)
    parallelize(game, params)
