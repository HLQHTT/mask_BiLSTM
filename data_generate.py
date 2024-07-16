import pandas as pd
import numpy as np
from dataset import Dataset, ESOL_Dataset
from dataset import vocab_generate

vocab = vocab_generate('。/datasets_deep_learning.csv', "SMILES表示")
# print(len(vocab))
# print(list(vocab.token_to_idx.items())[:38])

#提供给模型数据输入的接口，模型里只需调整种子的数字，即可得到不同的训练集与测试集
def data_generate(seed):
    #光敏分子数据集
    dataset = Dataset('./datasets_deep_learning.csv','SMILES表示','Φsam/Φref','溶剂SMILES', '标准品SMILES', '测定波长（nm）', 100,100,100,vocab,seed)
    data_smiles = dataset.get_data()
    x_train = data_smiles[0].astype('int32')
    y_train = data_smiles[1].astype('float32').reshape(-1, 1)
    solvent_train = data_smiles[6].astype('int32')
    ref_train = data_smiles[7].astype('int32')
    y_mean = y_train.mean()
    y_max = y_train.max()
    # y_min = y_train.min()
    y_train = (y_train-y_mean)/y_max
    # y_train = (y_train-y_min)/(y_max-y_min)

    x_test = data_smiles[2].astype('int32')
    y_test = data_smiles[3].astype('float32').reshape(-1, 1)
    y_test = (y_test-y_mean)/y_max
    solvent_test = data_smiles[8].astype('int32')
    ref_test = data_smiles[9].astype('int32')


    x_train_temp = pd.concat([pd.DataFrame(x_train), pd.DataFrame(solvent_train)], axis=1)
    x_train = np.array(pd.concat([x_train_temp, pd.DataFrame(ref_train)], axis=1)).astype('int32')
    x_test_temp = pd.concat([pd.DataFrame(x_test), pd.DataFrame(solvent_test)], axis=1)
    x_test = np.array(pd.concat([x_test_temp, pd.DataFrame(ref_test)], axis=1)).astype('int32')


    # pd.DataFrame(x_train).to_csv('./x_train.csv')
    # pd.DataFrame(y_train).to_csv('./y_train.csv')
    # pd.DataFrame(x_test).to_csv('./x_test.csv')
    # pd.DataFrame(y_test).to_csv('./y_test.csv')
#pd.DataFrame(train_temp).to_csv('./train_temp.csv')

    # data_smiles[4].to_csv('./train_set.csv')
    # data_smiles[5].to_csv('./test_set.csv')
    return x_train, y_train, x_test, y_test, y_mean, y_max

#x_train, y_train, x_test, y_test = data_generate(9)

def ESOL_data_generate(seed):
    #ESOL数据集
    dataset = ESOL_Dataset('./datasets/ESOL.csv','smiles','ESOL', 100,100,100,vocab,seed)
    #Lipophilicity数据集
    #dataset = ESOL_Dataset('./datasets/Lipophilicity.csv','smiles','exp', 100,100,100,vocab,seed)
    #SAMPL数据集
    #dataset = ESOL_Dataset('./datasets/SAMPL.csv','smiles','expt', 100,100,100,vocab,seed)
    #test_aug_times = dataset.test_augment_times
    #train_aug_times = dataset.train_augment_times
    data_smiles = dataset.get_data()
    x_train = data_smiles[0].astype('int32')
    y_train = data_smiles[1].astype('float32').reshape(-1, 1)
    y_mean = y_train.mean()
    y_max = y_train.max()
    y_train = (y_train-y_mean)/y_max

    x_test = data_smiles[2].astype('int32')
    y_test = data_smiles[3].astype('float32').reshape(-1, 1)
    y_test = (y_test-y_mean)/y_max

    return x_train, y_train, x_test, y_test