import pandas as pd
import numpy as np
from data_preprocess import randomize_smile
import re
from copy import deepcopy
import collections
regex_pattern=r'te|se|Cl|Br|Na|Te|Se|[./\\#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'


class Dataset(object):
    def __init__(self, filename,
                 smile_field,
                 label_field,
                 solvent_field,
                 ref_field,
                 wavelength_field,
                 max_len=100,
                 train_augment_times=1,
                 test_augment_times=1,
                 vocab=None,
                 random_state=0):

        df = pd.read_csv(filename
                         #, sep='\t'
                         )
        df['length'] = df[smile_field].map(lambda x: len(x.replace('Cl', 'X').replace('Br', 'Y').replace('Na', 'Z').replace('Te', 'W').replace('Se', 'U').replace('se', 'E').replace('te', 'G')))
        self.df = deepcopy(df[df.length <= max_len])
        self.smile_field = smile_field
        self.label_field = label_field
        self.solvent_field = solvent_field
        self.ref_field = ref_field
        self.wavelength_field = wavelength_field
        self.max_len = max_len
        self.train_augment_times = train_augment_times
        self.test_augment_times = test_augment_times
        self.random_state = random_state
        self.vocab =  vocab

    def numerical_smiles(self, data):
        x = np.zeros((len(data), (self.max_len + 2)), dtype='int32')
        y = np.array(data[self.label_field]).astype('float32')
        z = np.zeros((len(data), (self.max_len + 2)), dtype='int32')
        w = np.zeros((len(data), (self.max_len + 2)), dtype='int32')
        u = np.zeros((len(data), (8 + 2)), dtype='float32')
        for i,smiles in enumerate(data[self.smile_field].tolist()):
            smiles = self._char_to_idx(seq = smiles)
            smiles = self._pad_start_end_token(smiles)
            x[i,:len(smiles)] = np.array(smiles)
        for j,smiles in enumerate(data[self.solvent_field].tolist()):
            smiles = self._char_to_idx(seq = smiles)
            smiles = self._pad_start_end_token(smiles)
            z[j,:len(smiles)] = np.array(smiles)
        for k,smiles in enumerate(data[self.ref_field].tolist()):
            smiles = self._char_to_idx(seq = smiles)
            smiles = self._pad_start_end_token(smiles)
            w[k,:len(smiles)] = np.array(smiles)
        for l,wavelength in enumerate(data[self.wavelength_field].tolist()):
            wavelength = self._char_to_idx(seq = str(wavelength))
            wavelength = self._pad_start_end_token(wavelength)
            u[l,:len(wavelength)] = np.array(wavelength)
        return x, y, z, w, u
    

    def _pad_start_end_token(self,seq):
        seq.insert(0, self.vocab['<start>'])
        seq.append(self.vocab['<end>'])

        return seq

    def _char_to_idx(self,seq):
        char_list = re.findall(regex_pattern, seq)
        return [self.vocab[char_list[j]] for j in range(len(char_list))]
    def get_data(self):
        data = self.df
        length_count = data.length.value_counts()
        train_idx = []
        for k, v in length_count.items():
            if v >= 3:
                idx = data[data.length == k].sample(frac=0.8, random_state=self.random_state).index
            else:
                idx = data[data.length == k].sample(n=1, random_state=self.random_state).index
            train_idx.extend(idx)

        X_train = deepcopy(data[data.index.isin(train_idx)])
        X_test = deepcopy(data[~data.index.isin(train_idx)])
    
        if self.train_augment_times>1:
            train_temp = pd.concat([X_train] * (self.train_augment_times - 1), axis=0)
            train_temp[self.smile_field] = train_temp[self.smile_field].map(lambda x: randomize_smile(x))
            train_set = pd.concat([train_temp, X_train], ignore_index=True)
        else:
            train_set = X_train
        train_set.dropna(inplace=True)
        train_set = deepcopy(train_set)
        train_set['length'] = train_set[self.smile_field].map(lambda x: len(x.replace('Cl', 'X').replace('Br', 'Y').replace('Na', 'Z').replace('Te', 'W').replace('Se', 'U').replace('se', 'E').replace('te', 'G')))
        train_set = train_set[train_set.length <= self.max_len]

        if self.test_augment_times>1:
            test_temp = pd.concat([X_test] * (self.test_augment_times - 1), axis=0)
            test_temp[self.smile_field] = test_temp[self.smile_field].map(lambda x: randomize_smile(x))
            test_set = pd.concat([test_temp, X_test], ignore_index=True)
        else:
            test_set = X_test
        test_set = deepcopy(test_set)


        x_train, y_train, solvent_train, ref_train, wavelength_train = self.numerical_smiles(train_set)
        x_test, y_test, solvent_test, ref_test, wavelength_test = self.numerical_smiles(test_set)
        print(len(X_train)/len(X_train[self.smile_field].unique()))
        print(x_test.shape)
        return x_train, y_train, x_test, y_test, train_set, test_set, solvent_train, ref_train, solvent_test, ref_test, wavelength_train, wavelength_test

class ESOL_Dataset(object):
    def __init__(self, filename,
                 smile_field,
                 label_field,
                 max_len=100,
                 train_augment_times=1,
                 test_augment_times=1,
                 vocab=None,
                 random_state=0):

        df = pd.read_csv(filename
                         #, sep='\t'
                         )
        df['length'] = df[smile_field].map(lambda x: len(x.replace('Cl', 'X').replace('Br', 'Y').replace('Na', 'Z').replace('Te', 'W').replace('Se', 'U').replace('se', 'E').replace('te', 'G')))
        self.df = deepcopy(df[df.length <= max_len])
        self.smile_field = smile_field
        self.label_field = label_field
        self.max_len = max_len
        self.train_augment_times = train_augment_times
        self.test_augment_times = test_augment_times
        self.random_state = random_state
        self.vocab =  vocab

    def numerical_smiles(self, data):
        x = np.zeros((len(data), (self.max_len + 2)), dtype='int32')
        y = np.array(data[self.label_field]).astype('float32')
        for i,smiles in enumerate(data[self.smile_field].tolist()):
            smiles = self._char_to_idx(seq = smiles)
            smiles = self._pad_start_end_token(smiles)
            x[i,:len(smiles)] = np.array(smiles)
        return x, y
    

    def _pad_start_end_token(self,seq):
        seq.insert(0, self.vocab['<start>'])
        seq.append(self.vocab['<end>'])
        return seq

    def _char_to_idx(self,seq):
        char_list = re.findall(regex_pattern, seq)
        return [self.vocab[char_list[j]] for j in range(len(char_list))]
    
    def get_data(self):
        data = self.df
        length_count = data.length.value_counts()
        train_idx = []
        for k, v in length_count.items():
            if v >= 3:
                idx = data[data.length == k].sample(frac=0.8, random_state=self.random_state).index
            else:
                idx = data[data.length == k].sample(n=1, random_state=self.random_state).index
            train_idx.extend(idx)

        X_train = deepcopy(data[data.index.isin(train_idx)])
        X_test = deepcopy(data[~data.index.isin(train_idx)])
        if self.train_augment_times>1:
            train_temp = pd.concat([X_train] * (self.train_augment_times - 1), axis=0)
            train_temp[self.smile_field] = train_temp[self.smile_field].map(lambda x: randomize_smile(x))
            train_set = pd.concat([train_temp, X_train], ignore_index=True)
        else:
            train_set = X_train
        train_set.dropna(inplace=True)
        train_set = deepcopy(train_set)
        train_set['length'] = train_set[self.smile_field].map(lambda x: len(x.replace('Cl', 'X').replace('Br', 'Y').replace('Na', 'Z').replace('Te', 'W').replace('Se', 'U').replace('se', 'E').replace('te', 'G')))
        train_set = train_set[train_set.length <= self.max_len]

        if self.test_augment_times>1:
            test_temp = pd.concat([X_test] * (self.test_augment_times - 1), axis=0)
            test_temp[self.smile_field] = test_temp[self.smile_field].map(lambda x: randomize_smile(x))
            test_set = pd.concat([test_temp, X_test], ignore_index=True)
        else:
            test_set = X_test
        test_set = deepcopy(test_set)


        x_train, y_train = self.numerical_smiles(train_set)
        x_test, y_test = self.numerical_smiles(test_set)
        print(len(X_train)/len(X_train[self.smile_field].unique()))
        print(x_test.shape)
        return x_train, y_train, x_test, y_test, train_set, test_set

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
    data = pd.read_csv(data_path )
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
# [('<unk>', 0), ('c', 1), ('C', 2), ('(', 3), (')', 4), ('1', 5), 
#  ('2', 6), ('=', 7), ('3', 8), ('O', 9), ('[', 10), (']', 11), ('-', 12), 
#  ('N', 13), ('4', 14), ('n', 15), ('F', 16), ('+', 17), ('I', 18), ('Cl', 19), 
#  ('B', 20), ('Br', 21), ('5', 22), ('o', 23), ('H', 24), ('6', 25), ('S', 26), 
#  ('s', 27), ('7', 28), ('se', 29), ('te', 30), ('Te', 31), ('%', 32), ('0', 33), 
#  ('Se', 34), ('8', 35), ('9', 36), ('#', 37)] 
