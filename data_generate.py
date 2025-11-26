import pandas as pd
import numpy as np
from dataset import Dataset, ESOL_Dataset
from dataset import vocab_generate

vocab = vocab_generate('./datasets_deep_learning.csv', "SMILES表示")


#提供给模型数据输入的接口，模型里只需调整种子的数字，即可得到不同的训练集与测试集
def data_generate(seed):
    #光敏分子数据集
    dataset = Dataset('./datasets_deep_learning.csv','SMILES表示','Φsam/Φref','溶剂SMILES', '标准品SMILES', '测定波长（nm）', 100,100,100,vocab,seed)
    data_smiles = dataset.get_data()
    x_train = data_smiles[0].astype('int32')
    y_train = data_smiles[1].astype('float32').reshape(-1, 1)
    solvent_train = data_smiles[6].astype('int32')
    ref_train = data_smiles[7].astype('int32')
    mu_train = y_train.mean()
    std_train = y_train.std()
   
    y_train = (y_train - mu_train) / std_train
    

    x_test = data_smiles[2].astype('int32')
    y_test = data_smiles[3].astype('float32').reshape(-1, 1)
    y_test  = (y_test  - mu_train) / std_train
    solvent_test = data_smiles[8].astype('int32')
    ref_test = data_smiles[9].astype('int32')


    x_train_temp = pd.concat([pd.DataFrame(x_train), pd.DataFrame(solvent_train)], axis=1)
    x_train = np.array(pd.concat([x_train_temp, pd.DataFrame(ref_train)], axis=1)).astype('int32')
    x_test_temp = pd.concat([pd.DataFrame(x_test), pd.DataFrame(solvent_test)], axis=1)
    x_test = np.array(pd.concat([x_test_temp, pd.DataFrame(ref_test)], axis=1)).astype('int32')


    return x_train, y_train, x_test, y_test, y_mean, y_max


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
