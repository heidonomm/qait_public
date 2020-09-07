from kgdqn.models import KGDQN
from kgdqn.representations import StateNAction
from kgdqn.graph_replay import *
from kgdqn.graph_replay import GraphPriorityReplayBuffer, GraphReplayBuffer

from generic import list_of_token_list_to_char_input
from generic import to_np, to_pt, preproc, _words_to_ids, pad_sequences
from generic import max_len, ObservationPool
from layers import compute_mask
import qa_memory
import command_generation_memory

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import spacy

import logging


class Agent_KG:

    def __init__(self):
        self.mode = "train"
        self.config = self.get_params()

        self.model = KGDQN(self.config, "actions")

        self.num_episodes = self.config['num_episodes']
        self.state = StateNAction()

        self.update_freq = self.config['update_frequency']
        self.filename = 'kgdqn_' + \
            '_'.join([str(v)
                      for k, v in self.config.items() if 'file' not in str(k)])
        logging.basicConfig(filename='logs/' +
                            self.filename + '.log', filemode='w')
        logging.warning("Parameters", self.config)

        # tokenizer

        # not really used in the current model as capacity == 1
        self.naozi = ObservationPool(capacity=self.config['naozi_capacity'])

        if self.config['replay_buffer_type'] == 'priority':
            self.replay_buffer = GraphPriorityReplayBuffer(
                self.config['replay_buffer_size'])
        elif self.config['replay_buffer_type'] == 'standard':
            self.replay_buffer = GraphReplayBuffer(
                self.config['replay_buffer_size'])

        self.config["vocab_size"] = len(self.state.vocab_drqa)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config["lr"])

        self.rho = self.config["rho"]
        self.gamma = self.config["gamma"]

    def load_configuration(self):
        with open("vocabularies/word_vocab", encoding="utf-8") as f:
            self.word_vocab = f.read().split("\n")
        self.word2id = {index: word for word,
                        index in enumerate(self.word_vocab)}

        # char vocab
        with open("vocabularies/char_vocab.txt", encoding="utf-8") as f:
            self.char_vocab = f.read().split("\n")
        self.char2id = {}
        for i, w in enumerate(self.char_vocab):
            self.char2id[w] = i

        self.EOS_id = self.word2id["</s>"]
        # TODO: set all config params here
        self.question_type = "location"

        # Replay buffer and updates
        self.command_generation_replay_memory = command_generation_memory.PrioritizedReplayMemory(
            self.config["replay_buffer_size"])
        self.qa_replay_memory = qa_memory.PrioritizedReplayMemory(
            self.config["replay_buffer_size"])
        self.update_per_k_game_steps = self.config["update_freq"]

        # epsilon greedy
        self.epsilon_anneal_episodes = self.config["e_decay"]
        self.epsilon_anneal_from = self.config["e_start"]
        self.epsilon_anneal_to = self.config["e_final"]
        self.epsilon = self.epsilon_anneal_from

        # Set the random seed manually for reproducibility.
        self.random_seed = self.config['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print(
                    "WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False

        if self.question_type == "location":
            self.answer_type = "pointing"
        elif self.question_type in ["attribute", "existence"]:
            self.answer_type = "2 way"
        else:
            raise NotImplementedError

        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
        self.single_word_verbs = set(["inventory", "look", "wait"])
        self.two_word_verbs = set(["go"])

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.model.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """

        self.mode = "eval"

        self.model.eval()

    def save_model_to_path(self, save_to):
        torch.save(self.model.state_dict(), save_to)
        print(f"Saved checkpoint to {save_to}...")

    def init(self, obs, infos):
        """
        Prepare the agent for the upcoming games.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        batch_size = len(obs)
        self.reset_binarized_counter(batch_size)
        self.not_finished_yet = np.ones((batch_size,), dtype="float32")
        self.prev_actions = [["" for _ in range(batch_size)]]
        # 1s and starts to be 0 when previous action is "wait"
        self.prev_step_is_still_interacting = np.ones(
            (batch_size,), dtype="float32")
        self.naozi.reset(batch_size=batch_size)

    def get_agent_inputs(self, string_list):
        # split word wise
        sentence_token_list = [item.split() for item in string_list]
        # tokenize
        sentence_id_list = [_words_to_ids(
            tokens, self.word2id) for tokens in sentence_token_list]
        input_sentence = pad_sequences(
            sentence_id_list, maxlen=max_len(sentence_id_list)).astype("int32")
        input_sentence = to_pt(input_sentence, self.use_cuda)
        input_sentence_char = list_of_token_list_to_char_input(
            sentence_token_list, self.char2id)
        input_sentence_char = to_pt(input_sentence_char, self.use_cuda)

        return input_sentence, input_sentence_char, sentence_id_list

    def get_game_info_at_certain_step(self, obs, infos):
        """
        Get all needed info from game engine for training.
        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        batch_size = len(obs)
        feedback_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        description_strings = [preproc(item, tokenizer=self.nlp)
                               for item in infos["description"]]
        observation_strings = [desc + "<|> " + feedback if feedback != desc else desc +
                               " <|> hello" for feedback, desc in zip(feedback_strings, description_strings)]

        inventory_strings = [preproc(item, tokenizer=self.nlp)
                             for item in infos["inventory"]]

        local_word_list = [obs.split() + inv.split()
                           for obs, inv in zip(observation_strings, inventory_strings)]

        directions = ["east", "west", "north", "south"]
        if self.question_type in ["location", "existence"]:
            # agents observes the env, but do not change them
            possible_verbs = [["go", "inventory", "wait",
                               "open", "examine"] for _ in range(batch_size)]
        else:
            possible_verbs = [list(set(item) - set(["", "look"]))
                              for item in infos["verbs"]]

        possible_adjs, possible_nouns = [], []
        for i in range(batch_size):
            object_nouns = [item.split()[-1]
                            for item in infos["object_nouns"][i]]
            object_adjs = [w for item in infos["object_adjs"][i]
                           for w in item.split()]
            possible_nouns.append(list(set(object_nouns) & set(
                local_word_list[i]) - set([""])) + directions)
            possible_adjs.append(list(set(object_adjs) & set(
                local_word_list[i]) - set([""])) + ["</s>"])

        return observation_strings, [possible_verbs, possible_adjs, possible_nouns]

    def get_state_string(self, infos):
        description_strings = infos["description"]
        inventory_strings = infos["inventory"]
        observation_strings = [
            _d + _i for (_d, _i) in zip(description_strings, inventory_strings)]
        return observation_strings

    def get_local_word_masks(self, possible_words):
        possible_verbs, possible_adjs, possible_nouns = possible_words
        batch_size = len(possible_verbs)

        verb_mask = np.zeros(
            (batch_size, len(self.word_vocab)), dtype="float32")
        noun_mask = np.zeros(
            (batch_size, len(self.word_vocab)), dtype="float32")
        adj_mask = np.zeros(
            (batch_size, len(self.word_vocab)), dtype="float32")
        for i in range(batch_size):
            for w in possible_verbs[i]:
                if w in self.word2id:
                    verb_mask[i][self.word2id[w]] = 1.0
            for w in possible_adjs[i]:
                if w in self.word2id:
                    adj_mask[i][self.word2id[w]] = 1.0
            for w in possible_nouns[i]:
                if w in self.word2id:
                    noun_mask[i][self.word2id[w]] = 1.0
        adj_mask[:, self.EOS_id] = 1.0

        return [verb_mask, adj_mask, noun_mask]

    # TODO
    def get_match_representations(self, input_observation, input_observation_char, input_quest, input_quest_char, use_model="online"):
        description_representation_sequence, description_mask = self.model.representation_generator(
            input_observation, input_observation_char)
        quest_representation_sequence, quest_mask = self.model.representation_generator(
            input_quest, input_quest_char)
        match_representation_sequence = self.model.get_match_representations(
            description_representation_sequence, description_mask, quest_representation_sequence, quest_mask, self.state)

    # previously compute_td_loss
    def get_dqn_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size, self.rho)

        if self.use_cuda:
            reward = torch.FloatTensor(reward).cuda()
            done = torch.FloatTensor(1 * done).cuda()
            action_t = torch.LongTensor(action).cuda()
        else:
            reward = torch.FloatTensor(reward)
            done = torch.FloatTensor(1 * done)
            action_t = torch.LongTensor(action)

        q_value = self.model.forward_td_init(state, action_t)[0][0]

        with torch.no_grad():
            if self.use_cuda:
                actions = torch.LongTensor(
                    [a.pruned_actions_rep for a in list(next_state)]).cuda()
            else:
                actions = torch.LongTensor(
                    [a.pruned_actions_rep for a in list(next_state)])
            fwd_init, sts = self.model.forward_td_init(
                next_state, actions[:, 0, :])
            next_q_values = fwd_init[0].unsqueeze_(0)
            for i in range(1, actions.size(1)):
                act = actions[:, i, :]
                sts = sts.new_tensor(sts.data)
                cat_q = self.model.forward_td(sts, next_state, act)[
                    0].unsqueeze_(0)
                next_q_values = torch.cat((next_q_values, cat_q), dim=0)

            next_q_values = next_q_values.transpose(0, 1)

        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - (expected_q_value.data)).pow(2).mean()

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    def update_interaction(self):
        interaction_loss = self.get_dqn_loss()
        if interaction_loss is None:
            return None
        # # could call the following to prevent exploding gradient problem in RNNs / LSTMs (would require defining clip param)
        # torch.nn.utils.clip_grad_norm_(
        #   self.model.parameters(), self.clip_grad_norm)
        return to_np(interaction_loss)

    def answer_question(self, input_observation, input_observation_char, observation_id_list, input_quest, input_quest_char, use_model="online"):
        # first pad answerer input and get the mask
        batch_size = len(observation_id_list)
        max_length = input_observation.size(1)
        mask = compute_mask(input_observation)

        # noun_mask for location question
        if self.question_type in ["location"]:
            location_mask = []
            for i in range(batch_size):
                m = [1 for item in observation_id_list[i]]
                location_mask.append(m)
            location_mask = pad_sequences(
                location_mask, maxlen=max_length, dtype="float32")
            location_mask = to_pt(
                location_mask, enable_cuda=self.use_cuda, type="float")
            assert mask.size() == location_mask.size()
            mask = mask * location_mask

        match_representation_sequence = self.get_match_representations(
            input_observation, input_observation_char, input_quest, input_quest_char, use_model=use_model)

        pred = model.answer_question(match_representation_sequence, mask)
        if self.answer_type == "2 way":
            observation_id_list = []
            max_length = 2
            for i in range(batch_size):
                observation_id_list.append(
                    [self.word2id["0"], self.word2id["1"]])

        observation = to_pt(pad_sequences(
            observation_id_list, maxlen=max_length).astype("int32"), self.use_cuda)
        vocab_distribution = np.zeros(
            (batch_size, len(self.word_vocab)))
        vocab_distribution = to_pt(
            vocab_distribution, self.use_cuda, type='float')
        vocab_distribution = vocab_distribution.scatter_add_(
            1, observation, pred)  # batch x vocab
        non_zero_words = []
        for i in range(batch_size):
            non_zero_words.append(list(set(observation_id_list[i])))
        vocab_mask = torch.ne(vocab_distribution, 0).float()

        return vocab_distribution, non_zero_words, vocab_mask

    def finish_of_episode(self, episode_no, batch_size):
        # decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        if episode_no < self.e_decay + self.learn_start_from_this_episode:
            self.epsilon -= (self.epsilon_anneal_from -
                             self.epsilon_anneal_to) / float(self.epsilon_anneal_episodes)
            self.epsilon = max(self.epsilon, 0.0)

    def reset_binarized_counter(self, batch_size):
        self.binarized_counter_dict = [{} for _ in range(batch_size)]

    def get_binarized_count(self, observation_strings, update=True):
        count_rewards = []
        batch_size = len(observation_strings)
        for i in range(batch_size):
            key = observation_strings[i]
            if key not in self.binarized_counter_dict[i]:
                self.binarized_counter_dict[i][key] = 0.0
            if update:
                self.binarized_counter_dict[i][key] += 1.0
            r = self.binarized_counter_dict[i][key]
            r = float(r == 1.0)
            count_rewards.append(r)
        return count_rewards

    def set_episode_number(self, episode_count):
        self.episode_nr = episode_count

    def get_episode_number(self):
        return self.episode_nr

    def get_params(self):
        return {
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
            'e_start': 1.0,
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
            'doc_rnn_padding': True,
            'naozi_capacity': 1,
            'random_seed': 1234
        }
