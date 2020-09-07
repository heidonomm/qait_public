import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import spacy
import numpy as np
import random

from drqa import *
from layers import compute_mask, EncoderBlock, AnswerPointer, NoisyLinear, masked_softmax
from layers_kg import *


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class KGDQN(nn.Module):
    def __init__(self, params, actions):
        super(KGDQN, self).__init__()
        self.params = params

        # Check if cuda is available:
        self.use_cuda = True if torch.cuda.is_available() else False

        if self.params['qa_init']:
            pretrained_action_embs = torch.load(params['act_emb_init_file'])[
                'state_dict']['embeddings']['weight']
            self.action_emb = nn.Embedding.from_pretrained(
                pretrained_action_embs, freeze=False)
            self.action_drqa = ActionDrQA(params, pretrained_action_embs)
            self.state_gat = StateNetwork(
                actions, params, pretrained_action_embs)
        else:
            self.action_emb = nn.Embedding(
                params['vocab_size'], params['embedding_size'])
            self.action_drqa = ActionDrQA(params, self.action_emb.weight)
            self.state_gat = StateNetwork(
                actions, params, self.action_emb.weight)
        self.action_enc = EncoderLSTM(params['vocab_size'], params['embedding_size'], params['hidden_size'],
                                      params['padding_idx'], params['dropout_ratio'],
                                      self.action_emb)  # , params['bidirectional'],
        self.state_fc = nn.Linear(
            params['drqa_emb_size'] + params['hidden_size'], 100)

        # Addition
        # Question-Answerer Head for when the agent issues 'wait'
        self.aggregators = torch.nn.ModuleList([EncoderBlock(conv_num=self.params["aggregation_conv_num"], ch_num=self.params["block_hidden_dim"],
                                                             k=5, block_hidden_dim=self.params[
                                                                 "block_hidden_dim"], n_head=self.params["n_heads"],
                                                             dropout=self.params["block_dropout"]) for _ in range(self.params["aggregation_layers"])])

        linear_function = NoisyLinear if self.noisy_net else torch.nn.Linear

        self.answer_pointer = AnswerPointer(
            block_hidden_dim=self.params["block_hidden_dim"], noisy_net=self.params["noisy_net"])

        if self.answer_type in ["2 way"]:
            self.question_answerer_output_1 = linear_function(
                self.params["block_hidden_dim"], self.params["question_answerer_hidden_dim"])
            self.question_answerer_output_2 = linear_function(
                self.params["question_answerer_hidden_dim"], 2)

    def forward(self, s_t, emb_a_t, encoded_doc):
        batch = emb_a_t.size(0)
        state_emb = torch.cat((s_t, encoded_doc), dim=1)
        state_emb = self.state_fc(state_emb)
        q = torch.bmm(state_emb.view(batch, 1, 100),
                      emb_a_t.view(batch, 100, 1)).view(batch)
        return q, emb_a_t  # action_embedding

    def forward_td_init(self, state, a_t):
        state = list(state)

        drqa_input = torch.LongTensor(state[0].drqa_input).unsqueeze_(0).cuda(
        ) if self.use_cuda else torch.LongTensor(state[0].drqa_input).unsqueeze_(0)

        sts = self.state_gat(state[0].graph_state_rep).unsqueeze_(0)
        for i in range(1, len(state)):
            sts = torch.cat((sts, self.state_gat(
                state[i].graph_state_rep).unsqueeze_(0)), dim=0)
            drqa_input = torch.cat((drqa_input, torch.LongTensor(state[i].drqa_input).unsqueeze_(0).cuda(
            )), dim=0) if self.use_cuda else torch.cat((drqa_input, torch.LongTensor(state[i].drqa_input).unsqueeze_(0)), dim=0)

        encoded_doc = self.action_drqa(drqa_input, state)[1]

        _, emb_a_t, _ = self.action_enc(a_t)
        return self.forward(sts, emb_a_t, encoded_doc), sts  # .squeeze()

    def forward_td(self, state_rep, state, a_t):
        drqa_input = torch.LongTensor(state[0].drqa_input).unsqueeze_(0).cuda(
        ) if self.use_cuda else torch.LongTensor(state[0].drqa_input).unsqueeze_(0)
        for i in range(1, len(state)):
            drqa_input = torch.cat((drqa_input, torch.LongTensor(state[i].drqa_input).unsqueeze_(0).cuda(
            )), dim=0) if self.use_cuda else torch.cat((drqa_input, torch.LongTensor(state[i].drqa_input).unsqueeze_(0)), dim=0)
        encoded_doc = self.action_drqa(drqa_input, state)[1]
        _, emb_a_t, _ = self.action_enc(a_t)
        return self.forward(state_rep, emb_a_t, encoded_doc)

    def act(self, state, epsilon, epsilon2=0.15):

        graph_state_rep = state.graph_state_rep
        if not self.params['pruned']:
            epsilon2 = 0

        if random.random() > epsilon:

            feasible_actions_rep = state.all_actions_rep
            with torch.no_grad():
                drqa_input = torch.LongTensor(state.drqa_input).unsqueeze_(0).cuda(
                ) if self.use_cuda else torch.LongTensor(state.drqa_input).unsqueeze_(0)

                s_t = self.state_gat(graph_state_rep).unsqueeze_(0).repeat(len(feasible_actions_rep), 1).cuda(
                ) if self.use_cuda else self.state_gat(graph_state_rep).unsqueeze_(0).repeat(len(feasible_actions_rep), 1)

                encoded_doc = self.action_drqa(drqa_input, state)[1]
                # a_t = torch.LongTensor(feasible_actions_rep).cuda()#unsqueeze_(0).cuda()
                a_t = torch.LongTensor(feasible_actions_rep).cuda(
                ) if self.use_cuda else torch.LongTensor(feasible_actions_rep)

            encoded_doc = encoded_doc.repeat(len(feasible_actions_rep), 1)
            _, emb_a_t, _ = self.action_enc(a_t)

            fwd, fwd_at = self.forward(s_t, emb_a_t, encoded_doc)

            max_q, max_idx = torch.max(fwd, 0)

            action_ids = feasible_actions_rep[max_idx]
            picked = True
        else:
            if self.params['pruned']:
                if random.random() > epsilon2:
                    feasible_actions_rep = state.all_actions_rep
                else:
                    feasible_actions_rep = state.pruned_actions_rep
            else:
                feasible_actions_rep = state.pruned_actions_rep
            action_ids = feasible_actions_rep[random.randrange(
                len(feasible_actions_rep))]
            picked = False
        # , s_t[0].squeeze_(), fwd_at[max_idx].squeeze_()
        return action_ids, picked

    def representation_generator(self, _input_words, _input_chars):
        mask = compute_mask(_input_words)
        context, decoder_init, c_t = self.action_enc(_input_words)
        return decoder_init, mask

    # TODO: define epsilon, epsilon2
    def get_match_representations(self, doc_encodings, doc_mask, q_encodings, q_mask, state, epsilon, epsilon2=0.15):
        graph_state_rep = state.graph_state_rep
        if not self.params['pruned']:
            epsilon2 = 0

        if random.random() > epsilon:
            feasible_actions_rep = state.all_actions_rep
            with torch.no_grad():
                drqa_input = torch.LongTensor(state.drqa_input).unsqueeze_(0).cuda(
                ) if self.use_cuda else torch.LongTensor(state.drqa_input).unsqueeze_(0)

                s_t = self.state_gat(graph_state_rep).unsqueeze_(0).repeat(len(feasible_actions_rep), 1).cuda(
                ) if self.use_cuda else self.state_gat(graph_state_rep).unsqueeze_(0).repeat(len(feasible_actions_rep), 1)

                encoded_doc = self.action_drqa(drqa_input, state)[1]
                a_t = torch.LongTensor(feasible_actions_rep).cuda(
                ) if self.use_cuda else torch.LongTensor(feasible_actions_rep)

            encoded_doc = encoded_doc.repeat(len(feasible_actions_rep), 1)
            _, emb_a_t, _ = self.action_enc(a_t)

            fwd, fwd_at = self.forward(s_t, emb_a_t, encoded_doc)

            max_q, max_idx = torch.max(fwd, 0)

            action_ids = feasible_actions_rep[max_idx]
            picked = True
        else:
            if self.params['pruned']:
                if random.random() > epsilon2:
                    feasible_actions_rep = state.all_actions_rep
                else:
                    feasible_actions_rep = state.pruned_actions_rep
            else:
                feasible_actions_rep = state.pruned_actions_rep
            actions_ids = feasible_actions_rep[random.randrange(
                len(feasible_actions_rep))]
            picked = False

        return actions_ids, picked

    def answer_question(self, matching_representation_sequence, doc_mask):
        square_mask = torch.bmm(doc_mask.unsqueeze(-1),
                                doc_mask.unsquueze(1))  # batch x time x time

        M0 = matching_representation_sequence
        M1 = M0
        for i in range(self.aggregation_layers):
            M0 = self.aggregators[i](
                M0, doc_mask, square_mask, i *
                (self.params["aggregation_conv_num"] + 2) +
                1, self.params["aggregation_layers"]
            )
        M2 = M0
        pred = self.answer_pointer(M1, M2, doc_mask)  # batch x time
        pred_distribution = masked_softmax(pred, m=doc_mask, axis=-1)
        if self.answer_type == "pointing":
            return pred_distribution

        z = torch.bmm(pred_distribution.view(pred_distribution.size(
            0), 1, pred_distribution.size(1)), M2)
        z = z.view(z.size(0), -1)  # batch x inp
        hidden = self.question_answerer_output_1(z)  # batch x hid
        hidden = torch.relu(hidden)  # batch x hid
        pred = self.question_answerer_output_2(hidden)  # batch x out
        pred = masked_softmax(pred, axis=-1)
        return pred


class StateNetwork(nn.Module):
    # embeddings == embeddings_weights
    def __init__(self, action_set, params, embeddings=None):
        super(StateNetwork, self).__init__()

        # Check if cuda is available:
        self.use_cuda = True if torch.cuda.is_available() else False

        self.params = params
        self.action_set = action_set
        self.vocab_kge = self.load_vocab_kge()
        self.vocab = self.load_vocab()
        self.gat = GAT(params['gat_emb_size'], 3, len(
            action_set), params['dropout_ratio'], 0.2, 1)

        if params['qa_init']:
            self.pretrained_embeds = nn.Embedding.from_pretrained(
                embeddings, freeze=False)
        else:
            self.pretrained_embeds = nn.Embedding(
                len(self.vocab), params["embedding_size"], freeze=False)
            # self.pretrained_embeds = embeddings.new_tensor(embeddings.data)

        self.init_state_ent_emb()
        self.fc1 = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, 100)

    def init_state_ent_emb(self):
        embeddings = torch.zeros(
            (len(self.vocab_kge), self.params['embedding_size']))
        for i in range(len(self.vocab_kge)):
            graph_node_text = self.vocab_kge[i].split('_')
            graph_node_ids = []
            for w in graph_node_text:
                if w in self.vocab.keys():
                    if self.vocab[w] < len(self.vocab) - 2:
                        graph_node_ids.append(self.vocab[w])
                    else:
                        graph_node_ids.append(1)
                else:
                    graph_node_ids.append(1)
            graph_node_ids = torch.LongTensor(graph_node_ids).cuda(
            ) if torch.cuda.is_available() else torch.LongTensor(graph_node_ids)
            cur_embeds = self.pretrained_embeds(graph_node_ids)

            cur_embeds = cur_embeds.mean(dim=0)
            embeddings[i, :] = cur_embeds
        self.state_ent_emb = nn.Embedding.from_pretrained(
            embeddings, freeze=False)

    def load_vocab_kge(self):
        ent = {}
        with open('./entity2id.tsv', 'r') as f:
            for line in f:
                e, eid = line.split('\t')
                ent[int(eid.strip())] = e.strip()
        return ent

    def load_vocab(self):
        vocab_dict = {}
        with open("./w2id.txt", encoding="utf-8") as f:
            vocab_list = f.read().split()
        for i, word in enumerate(vocab_list):
            vocab_dict[word] = i
        return vocab_dict

    def forward(self, graph_rep):
        node_feats, adj = graph_rep
        adj = torch.IntTensor(adj).cuda(
        ) if self.use_cuda else torch.IntTensor(adj)

        x = self.gat(self.state_ent_emb.weight, adj).view(-1)
        out = self.fc1(x)
        return out


class ActionDrQA(nn.Module):
    def __init__(self, opt, embeddings):
        super(ActionDrQA, self).__init__()
        doc_input_size = opt['embedding_size']

        # Check if cuda is available:
        self.use_cuda = True if torch.cuda.is_available() else False

        if opt['qa_init']:
            self.embeddings = nn.Embedding.from_pretrained(
                embeddings, freeze=False)
        else:
            self.embeddings = nn.Embedding(
                opt['vocab_size'], opt['embedding_size'])
        self.doc_rnn = StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['doc_hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['doc_dropout_rnn'],
            dropout_output=opt['doc_dropout_rnn_output'],
            concat_layers=opt['doc_concat_rnn_layers'],
            rnn_type=nn.LSTM,
            padding=opt['doc_rnn_padding'],
        )
        if opt['qa_init']:
            inter = torch.load(opt['act_emb_init_file'])[
                'state_dict']['doc_encoder']  # ['weight']
            self.doc_rnn.load_state_dict(inter)

    def forward(self, vis_state_tensor, state):
        mask = torch.IntTensor([80] * vis_state_tensor.size(0)).cuda(
        ) if self.use_cuda else torch.IntTensor([80] * vis_state_tensor.size(0))

        emb_tensor = self.embeddings(vis_state_tensor)
        return self.doc_rnn(emb_tensor, mask)
