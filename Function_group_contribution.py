from substructure_generate import get_fg_index
from BiLSTM import predict_fuc
import pandas as pd
import re
import numpy as np
from data_generate import vocab
from data_generate import data_generate
from copy import deepcopy
from rdkit import Chem
from tensorflow import keras
from Substructure_contribution import re_sort, remove_mask_index
from rdkit.Chem.SaltRemover import SaltRemover
from data_preprocess import remove_salt_stereo
from Substructure_contribution import return_prediction, ESOL_return_prediction
REMOVER = SaltRemover()
regex_pattern=r'te|se|Cl|Br|Na|Te|Se|[./\\#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'
max_len=100
batch_size = 64
epochs = 50
#mol是要分析的分子index
#mol = 32
char = ['C', 'c', 'O', 'N', 'F', 'n', 'I', 'Cl', 'S', 'B', 'Br', 's', 'P', 'Na', 'Te', 'Se', 'o', 'se', 'te']
number = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']



#在list开始处和结尾处插入'<start>'和'<end>'，在这里也就是插入0值
def _pad_start_end_token(seq):
    seq.insert(0, vocab['<start>'])
    seq.append(vocab['<end>'])
    return seq

def canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)  # 使用RDKit MolFromSmiles函数标准化SMILES
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)  # 获取标准化后的SMILES
        return canonical_smiles
    except:
        return None
    


#读入数据，其中smiles代表所有分子的SMILES，solvent为溶剂SMILES，ref为参比物的SMILES，y_label是标签（激发波长为509 nm）    
data = pd.read_csv('./datasets_deep_learning.csv')
smiles = data['SMILES表示']
solvent = data['溶剂SMILES']
ref = data['标准品SMILES']
y_label = data['Φsam/Φref']
_, __, ___, ____, y_mean, y_max = data_generate(39)

ESOL_data = pd.read_csv('./ESOL.csv')
ESOL_smiles = ESOL_data['smiles']
ESOL_y_label = ESOL_data['ESOL']

smile_canonical = []
for smile in smiles:
    smile_canonical.append(canonical_smiles(smile))

smile_canonical_remove_salt = []
for smile in smile_canonical:
    smile_canonical_remove_salt.append(remove_salt_stereo(smile, REMOVER))

ESOL_smile_canonical = []
for smile in ESOL_smiles:
    ESOL_smile_canonical.append(canonical_smiles(smile))

ESOL_smile_canonical_remove_salt = []
for smile in ESOL_smile_canonical:
    ESOL_smile_canonical_remove_salt.append(remove_salt_stereo(smile, REMOVER))


#模型的建立
# xtrain, ytrain, xtest, ytest = data_generate(9)
# model = predict_fuc(batch_size=batch_size, epochs=epochs, xtrain=xtrain, ytrain=ytrain)
#model = keras.models.load_model("./deep learning/mask_BiLSTM/model/ESOL_BiLSTM_attention.h5", compile = False)
#光敏分子模型
# model = keras.models.load_model("./deep learning/mask_BiLSTM/model/BiLSTM_attention.h5", compile = False)
model_1 = keras.models.load_model(".model/BiLSTM_attention_1.h5", compile = False)
model_2 = keras.models.load_model("./model/BiLSTM_attention_2.h5", compile = False)
model_3 = keras.models.load_model("./model/BiLSTM_attention_3.h5", compile = False)
model_4 = keras.models.load_model("./model/BiLSTM_attention_4.h5", compile = False)
model_5 = keras.models.load_model("./model/BiLSTM_attention_5.h5", compile = False)
model_6 = keras.models.load_model("./model/BiLSTM_attention_6.h5", compile = False)
model_7 = keras.models.load_model("./model/BiLSTM_attention_7.h5", compile = False)
model_8 = keras.models.load_model("./model/BiLSTM_attention_8.h5", compile = False)
model_9 = keras.models.load_model("./model/BiLSTM_attention_9.h5", compile = False)
model_10 = keras.models.load_model("./model/BiLSTM_attention_10.h5", compile = False)

#ESOL模型
# model_1 = keras.models.load_model("./model/ESOL_BiLSTM_attention_9.h5", compile = False)
# model_2 = keras.models.load_model("./model/ESOL_BiLSTM_attention_29.h5", compile = False)
# model_3 = keras.models.load_model("./model/ESOL_BiLSTM_attention_49.h5", compile = False)
# model_4 = keras.models.load_model("./model/ESOL_BiLSTM_attention_69.h5", compile = False)
# model_5 = keras.models.load_model("./model/ESOL_BiLSTM_attention_89.h5", compile = False)
# model = tf.keras.models.load_model('./model/BiLSTM_attention')
print("-----模型加载完成-----")


def return_contribution(mol): 
    #将溶剂SMILES、参比物SMILES分成一个一个字符
    solvent_list = re.findall(regex_pattern, solvent[mol])
    ref_list = re.findall(regex_pattern, ref[mol])

    #分别创建溶剂，参比物的array矩阵，长度为max_len+2
    solvent_index = np.zeros((max_len + 2), dtype='int32')
    ref_index = np.zeros((max_len + 2), dtype='int32')

    #将溶剂与参比物的SMILES编码为数字
    solvent_num_list = [vocab[solvent_list[k]] for k in range(len(solvent_list))]
    solvent_num_list = _pad_start_end_token(solvent_num_list)
    ref_num_list = [vocab[ref_list[k]] for k in range(len(ref_list))]
    ref_num_list = _pad_start_end_token(ref_num_list)

    solvent_index[0:len(solvent_num_list)] = np.array(solvent_num_list)
    ref_index[0:len(ref_num_list)] = np.array(ref_num_list)
    #真实标签（标准化）
    y_true = (return_prediction(mol) - y_mean) / y_max
    #得到分子所有的原子序号列表
    mo = Chem.MolFromSmiles(smile_canonical_remove_salt[mol])
    mo_list = range(mo.GetNumAtoms())
    #得到所有的function group序列
    fg_index, fg_name = get_fg_index(smile_canonical_remove_salt[mol])

    if fg_index is not None:
        fg_contribution = []
        for m in range(len(fg_index)): #遍历所有的fg片段
            number_fg = []
            for n in range(len(fg_index[m])):
                for k in fg_index[m][n]:
                    number_fg.append(k)
            #smile_list = re_sort(number_fg, smile_canonical_remove_salt[mol])
            remove_bond_list = remove_mask_index(mo_list, number_fg)
            if remove_bond_list == []:
                smile_list = Chem.MolToSmiles(mo)
            else:
                smile_list = Chem.MolFragmentToSmiles(mo, atomsToUse=remove_bond_list)
            #得到掩盖掉function group并且数字化的list
            smile_num_list = [vocab[smile_list[k]] for k in range(len(smile_list))]
            #加入开始的0和结尾的0
            smile_num_list = _pad_start_end_token(smile_num_list)
            #将数字化之后的list填充到smiles_index的array中
            smiles_index = np.zeros((max_len + 2), dtype='int32')
            smiles_index[0:len(smile_num_list)] = np.array(smile_num_list)
            x_train = pd.DataFrame(np.hstack((smiles_index, solvent_index, ref_index))).T
            #y_pred = model.predict(x_train)
            y_pred = model_1.predict(x_train).tolist()[0][0]
            for model in [model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10]:
                y_pred += model.predict(x_train).tolist()[0][0]
            y_pred /= 10
            fg_contribution.append(y_true-y_pred)

        fg_pd = pd.DataFrame(fg_contribution).T
        fg_pd.columns = fg_name
        fg_pd.to_csv("./data/FG_contribution/fg_contributin of mol_{}.csv".format(mol))
        print("-----mol_{}结果已输出-----".format(mol))



def ESOL_return_contribution(mol): 
    #真实标签（标准化）
    y_true = ESOL_return_prediction(mol)
    #得到分子所有的原子序号列表
    mo = Chem.MolFromSmiles(ESOL_smile_canonical_remove_salt[mol])
    mo_list = range(mo.GetNumAtoms())
    #得到所有的function group序列
    fg_index, fg_name = get_fg_index(ESOL_smile_canonical_remove_salt[mol])

    if fg_index is not None:
        fg_contribution = []
        for m in range(len(fg_index)): #遍历所有的fg片段
            number_fg = []
            for n in range(len(fg_index[m])):
                for k in fg_index[m][n]:
                    number_fg.append(k)
            remove_bond_list = remove_mask_index(mo_list, number_fg)
            if remove_bond_list == []:
                smile_list = Chem.MolToSmiles(mo)
            else:
                smile_list = Chem.MolFragmentToSmiles(mo, atomsToUse=remove_bond_list)
            
            #得到掩盖掉function group并且数字化的list
            smile_num_list = [vocab[smile_list[k]] for k in range(len(smile_list))]
            #加入开始的0和结尾的0
            smile_num_list = _pad_start_end_token(smile_num_list)
            #将数字化之后的list填充到smiles_index的array中
            smiles_index = np.zeros((max_len + 2), dtype='int32')
            smiles_index[0:len(smile_num_list)] = np.array(smile_num_list)
    
            x_train = pd.DataFrame(np.hstack((smiles_index))).T
            #y_pred = model.predict(x_train)
            y_pred = (model_1.predict(x_train) + model_2.predict(x_train) + model_3.predict(x_train) + model_4.predict(x_train) + model_5.predict(x_train)) / 5
            fg_contribution.append(y_true-y_pred.tolist()[0][0])

        fg_pd = pd.DataFrame(fg_contribution).T
        fg_pd.columns = fg_name
        fg_pd.to_csv("./deep learning/mask_BiLSTM/data/ESOL/fg_contributin of mol_{}.csv".format(mol))
        print("-----mol_{}结果已输出-----".format(mol))
    # return atom_contribution, atom_list, brics_atom_contribution, bond_contribution, brics_substructure_contribution, brics_substructure_list
    
# atom_contribution, atom_list, brics_bond_contribution, brics_bond_list, brics_substructure_contribution, brics_substructure_list = return_contribution()

# pd.DataFrame(atom_contribution).to_csv('./atom_contribution_{}.csv'.format(mol))
for i in range(len(ESOL_smile_canonical_remove_salt)):
    if len(ESOL_smile_canonical_remove_salt[i]) <= max_len:
        ESOL_return_contribution(i)