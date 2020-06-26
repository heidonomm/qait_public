import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertConfig
import yaml
import torch

import warnings
from model import DQN

warnings.simplefilter(action='ignore', category=FutureWarning)

vocab = None
with open("vocabularies/word_vocab.txt") as f:
    vocab = f.read().split("\n")

with open("config.yaml") as reader:
    config = yaml.safe_load(reader)

vocab_len = len(vocab)

model = DQN(config, 30524)

total_params = sum(p.numel() for p in model.parameters())
print(total_params)

# print(model)


def compute_mask(x):
    mask = torch.ne(x, 0).float()
    if x.is_cuda:
        mask = mask.cuda()
    return mask


# arr = ['where is the red kartul?</s>']

# # string = "American actor did that - because that's the way things are."
# # question = "Which countertop is the actor on"

# # special_tokens = ["!", '"', "$$", "'", ",", "-=", ".",
# #                   "/", ":", ";", "=-", "=", "?", "`", "(", ")", "a-"]
# # tokenizer = DistilBertTokenizerFast(
# #     "vocabularies/word_vocab.txt", bos_token="<s>", eos_token="</s>", unk_token="<unk>", sep_token="<|>", pad_token="<pad>",
# #     additional_special_tokens=special_tokens)


# tokenizer = DistilBertTokenizerFast.from_pretrained(
#     'distilbert-base-uncased')
# tokenizer.add_tokens(['</s>'])

# print(tokenizer.decode([12612]))

# answer = [tokenizer.tokenize(sentence, add_special_tokens=False)
#           for sentence in arr]


# answer = [tokenizer.convert_tokens_to_ids(token) for token in answer]
# print(torch.Tensor(answer))
# mask = compute_mask(torch.Tensor(answer))

# model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# model.resize_token_embeddings(len(tokenizer))
# print(len(model.embeddings.word_embeddings.weight[-1]))


# print(model(torch.LongTensor(answer), attention_mask=mask))
