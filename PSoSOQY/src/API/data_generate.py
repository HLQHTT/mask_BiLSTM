# import torch
# import torch.nn as nn
# import torch.optim as optim
import pandas as pd
import numpy as np
import collections
import re
regex_pattern=r'te|se|Cl|Br|Na|Te|Se|[./\\#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens): 
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 词元化。每个文本序列被拆分成一个词元列表，词元（token）是文本的基本单位。 最后，返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（string）
def vocab_generate(data_path, smiles_field):
    data = pd.read_csv(#'D:\桌面\project\datasets\datasets_244.csv'
                         #, sep='\t'
                         data_path
                         )
    SMILES = data[
        #"SMILES表示"
        smiles_field
        ]
    tokens = []
    for i in range(len(SMILES)):
        char_list = re.findall(regex_pattern, SMILES[i])
        #char_list = list(map(list, zip(*char_list)))
        tokens.append(char_list)
    vocab = Vocab(tokens)
    return vocab
#np.save('./vocab.npy', vocab)
#print(list(vocab.token_to_idx.items())[:39])
#[('<unk>', 0), ('C', 1), ('=', 2), ('(', 3), (')', 4), 
# ('c', 5), ('1', 6), ('2', 7), ('O', 8), ('3', 9), ('4', 10), 
# ('N', 11), ('[', 12), (']', 13), ('F', 14), ('5', 15), ('-', 16), 
# ('n', 17), ('/', 18), ('+', 19), ('\\', 20), ('I', 21), ('6', 22), 
# ('Cl', 23), ('S', 24), ('B', 25), ('7', 26), ('.', 27), ('Br', 28), 
# ('H', 29), ('%', 30), ('8', 31), ('9', 32), ('0', 33), ('s', 34), 
# ('P', 35), ('Na', 36), ('Te', 37), ('#', 38), ('Se', 39), ('@', 40), ('o', 41)]


vocab = vocab_generate('D:\桌面\project\datasets\datasets_244.csv', "SMILES表示")
#print(len(vocab))
#print(list(vocab.token_to_idx.items())[:43])

