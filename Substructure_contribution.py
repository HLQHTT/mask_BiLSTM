from substructure_generate import get_mask_index
from rdkit.Chem import BRICS
from BiLSTM import predict_fuc
import pandas as pd
import re
import numpy as np
from data_generate import vocab
from data_generate import data_generate
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from data_preprocess import remove_salt_stereo
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from rdkit.Chem import BRICS
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from substructure_generate import get_fg_index
import base64
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 隐藏 GPU 设备
REMOVER = SaltRemover()
regex_pattern=r'te|se|Cl|Br|Na|Te|Se|[./\\#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'
max_len=100
# batch_size = 64
# epochs = 20
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
    
def get_bond_index_list(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)  # 使用RDKit MolFromSmiles函数标准化SMILES
        bond_index_list = []
        for i in range(mol.GetNumBonds()):
            bond = mol.GetBondWithIdx(i)
            bond_idx = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            bond_index_list.append(bond_idx)
        return bond_index_list
    except:
        return None


#读入数据，其中smiles代表所有分子的SMILES，solvent为溶剂SMILES，ref为参比物的SMILES，y_label是标签（激发波长为509 nm）    
data = pd.read_csv('./datasets_deep_learning.csv')
smiles = data['SMILES表示']
solvent = data['溶剂SMILES']
ref = data['标准品SMILES']
y_label = data['Φsam/Φref']
_, __, ___, ____, y_mean, y_max = data_generate(39)

ESOL_data = pd.read_csv('./SAMPL.csv')
ESOL_smiles = ESOL_data['smiles']
ESOL_y_label = ESOL_data['expt']

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
# model_1 = keras.models.load_model("./deep learning/mask_BiLSTM/model/SAMPL_BiLSTM_attention_9.h5", compile = False)
# model_2 = keras.models.load_model("./deep learning/mask_BiLSTM/model/SAMPL_BiLSTM_attention_29.h5", compile = False)
# model_3 = keras.models.load_model("./deep learning/mask_BiLSTM/model/SAMPL_BiLSTM_attention_49.h5", compile = False)
# model_4 = keras.models.load_model("./deep learning/mask_BiLSTM/model/SAMPL_BiLSTM_attention_69.h5", compile = False)
# model_5 = keras.models.load_model("./deep learning/mask_BiLSTM/model/SAMPL_BiLSTM_attention_89.h5", compile = False)
print("-----模型加载完成-----")


#去掉需mask的子结构index
def remove_mask_index(list1, list2):
    #原始列表
    original_list = list1
    #要删除的元素列表
    elements_to_remove = list2
    #使用列表推导式创建一个新列表，只包含不在要删除的元素列表中的原始列表元素
    filtered_list = [x for x in original_list if x not in elements_to_remove]
    
    return filtered_list



#对分子原有SMILES进行重排，去掉需要mask的子结构的SMILES，返回最终的SMILES
def re_sort(index_list, smile):

    #将SMILES转换成smile_list
    smile_list = re.findall(regex_pattern, smile)
    #保存smile_list中的字符的位置信息
    char_index = []
    for i in range(len(smile_list)):
        if smile_list[i] in char:
            char_index.append(i)
    #去掉smile_list中的index_list中的字符,并在最后添加'unk'
    elements_to_remove = []
    for i in index_list:
        elements_to_remove.append(char_index[i])
        if char_index[i]+1 < len(smile_list) and smile_list[char_index[i]+1] == '=':
            elements_to_remove.append(char_index[i]+1)
        if char_index[i]-1 >= 0 and smile_list[char_index[i]-1] == '=':
            elements_to_remove.append(char_index[i]-1)
    filtered_list = []
    for i in range(len(smile_list)):
        if i not in elements_to_remove:
            filtered_list.append(smile_list[i])
    #返回最后的smile_list
    return filtered_list

def return_prediction(mol):
    #将分子SMILES以及溶剂SMILES、参比物SMILES分成一个一个字符
    smile_list = re.findall(regex_pattern, smile_canonical_remove_salt[mol])
    solvent_list = re.findall(regex_pattern, solvent[mol])
    ref_list = re.findall(regex_pattern, ref[mol])
    #创建array矩阵，长度为max_len+2
    solvent_index = np.zeros((max_len + 2), dtype='int32')
    ref_index = np.zeros((max_len + 2), dtype='int32')
    smiles_index = np.zeros((max_len + 2), dtype='int32')

    #将SMILES编码为数字
    solvent_num_list = [vocab[solvent_list[k]] for k in range(len(solvent_list))]
    solvent_num_list = _pad_start_end_token(solvent_num_list)
    ref_num_list = [vocab[ref_list[k]] for k in range(len(ref_list))]
    ref_num_list = _pad_start_end_token(ref_num_list)
    smile_num_list = [vocab[smile_list[k]] for k in range(len(smile_list))]
    smile_num_list = _pad_start_end_token(smile_num_list)

    #将数字化之后的list填充到array中
    solvent_index[0:len(solvent_num_list)] = np.array(solvent_num_list)
    ref_index[0:len(ref_num_list)] = np.array(ref_num_list)
    smiles_index[0:len(smile_num_list)] = np.array(smile_num_list)
    
    x_train = pd.DataFrame(np.hstack((smiles_index, solvent_index, ref_index))).T
    y_pred = (model_1.predict(x_train).tolist()[0][0] * y_max) + y_mean
    for model in [model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10]:
            y_pred += (model.predict(x_train).tolist()[0][0] * y_max) + y_mean
    y_pred /= 10
    return y_pred

def ESOL_return_prediction(mol):
    #将分子SMILES以及溶剂SMILES、参比物SMILES分成一个一个字符
    smile_list = re.findall(regex_pattern, ESOL_smile_canonical_remove_salt[mol])
    #创建array矩阵，长度为max_len+2
    smiles_index = np.zeros((max_len + 2), dtype='int32')

    #将SMILES编码为数字

    smile_num_list = [vocab[smile_list[k]] for k in range(len(smile_list))]
    smile_num_list = _pad_start_end_token(smile_num_list)

    #将数字化之后的list填充到array中
    smiles_index[0:len(smile_num_list)] = np.array(smile_num_list)
    
    x_train = pd.DataFrame(smiles_index).T
    y_pred = model_1.predict(x_train).tolist()[0][0]
    for model in [model_2, model_3, model_4, model_5]:
            y_pred += model.predict(x_train).tolist()[0][0]
    y_pred /= 5
    return y_pred

def return_contribution(mol): 
    #将溶剂SMILES、参比物SMILES分成一个一个字符
    solvent_canonical = canonical_smiles(solvent[mol])
    solvent_canonical_remove_salt = remove_salt_stereo(solvent_canonical, REMOVER)
    ref_canonical = canonical_smiles(ref[mol])
    ref_canonical_remove_salt = remove_salt_stereo(ref_canonical, REMOVER)
    solvent_list = re.findall(regex_pattern, solvent_canonical_remove_salt)
    ref_list = re.findall(regex_pattern, ref_canonical_remove_salt)

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
    #真实标签(标准化)
    # y_true = (y_label[mol] - y_mean) / y_max
    y_true = (return_prediction(mol) - y_mean) / y_max
    # y_true = (return_prediction(mol) - y_min) / (y_max - y_min)
    #得到所有的BRICS片段序列
    mask_index = get_mask_index(smile_canonical_remove_salt[mol])
    #得到所有键的index
    bond_index = get_bond_index_list(smile_canonical_remove_salt[mol]) 
    #得到分子所有的原子序号列表
    mo = Chem.MolFromSmiles(smile_canonical_remove_salt[mol])
    mol_list = range(mo.GetNumAtoms())
    #键的贡献度计算
    bond_contribution = []
    for m in range(len(bond_index)): #遍历所有的键
        remove_bond_list = remove_mask_index(mol_list, bond_index[m])
        smile_list = Chem.MolFragmentToSmiles(mo, atomsToUse=remove_bond_list)
        
        #得到掩盖掉键并且数字化的list
        smile_num_list = [vocab[smile_list[k]] for k in range(len(smile_list))]
        #加入开始的0和结尾的0
        smile_num_list = _pad_start_end_token(smile_num_list)
        #将数字化之后的list填充到smiles_index的array中
        smiles_index = np.zeros((max_len + 2), dtype='int32')
        smiles_index[0:len(smile_num_list)] = np.array(smile_num_list)
    
        x_train = pd.DataFrame(np.hstack((smiles_index, solvent_index, ref_index))).T
        y_pred = model_1.predict(x_train).tolist()[0][0]
        # y_pred = (((model_1.predict(x_train).tolist()[0][0] * y_max) + y_mean) - y_min) / (y_max - y_min)
        for model in [model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10]:
            y_pred += model.predict(x_train).tolist()[0][0]
            # y_pred += (((model.predict(x_train).tolist()[0][0] * y_max) + y_mean) - y_min) / (y_max - y_min)
        y_pred /= 10
        bond_contribution.append(y_true-y_pred) 
    #BRICS片段贡献度计算
    atom_list = []
    brics_bond_list = []
    brics_substructure_list = []
    atom_contribution = []
    brics_bond_contribution = []
    brics_substructure_contribution = []
    for m in range(len(mask_index)): #遍历所有的BRICS片段
        #smile_list = re_sort(mask_index[m], smile_canonical_remove_salt[mol])
        remove_bond_list = remove_mask_index(mol_list, mask_index[m])
        if remove_bond_list == []:
            smile_list = Chem.MolToSmiles(mo)
        else:
            smile_list = Chem.MolFragmentToSmiles(mo, atomsToUse=remove_bond_list)
    
        #得到掩盖掉BRICS片段并且数字化的list
        smile_num_list = [vocab[smile_list[k]] for k in range(len(smile_list))]
        #加入开始的0和结尾的0
        smile_num_list = _pad_start_end_token(smile_num_list)
        #将数字化之后的list填充到smiles_index的array中
        smiles_index = np.zeros((max_len + 2), dtype='int32')
        smiles_index[0:len(smile_num_list)] = np.array(smile_num_list)
    
        x_train = pd.DataFrame(np.hstack((smiles_index, solvent_index, ref_index))).T
        #取10个模型的平均值
        y_pred = model_1.predict(x_train).tolist()[0][0]
        # y_pred = (((model_1.predict(x_train).tolist()[0][0] * y_max) + y_mean) - y_min) / (y_max - y_min)
        for model in [model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10]:
            y_pred += model.predict(x_train).tolist()[0][0]
            # y_pred += (((model.predict(x_train).tolist()[0][0] * y_max) + y_mean) - y_min) / (y_max - y_min)
        y_pred /= 10
        if len(mask_index[m]) == 1:
            atom_contribution.append(y_true-y_pred)
            atom_list.append(mask_index[m])
        elif len(mask_index[m]) == 2:
            brics_bond_contribution.append(y_true-y_pred)
            brics_bond_list.append(sorted(mask_index[m]))
        else:
            brics_substructure_contribution.append(y_true-y_pred)
            mask_index_sorted = sorted(mask_index[m])
            brics_substructure_list.append(mask_index_sorted)

    for m in range(len(brics_substructure_list)):
            for i in range(len(bond_index)):
                if bond_index[i][0] in brics_substructure_list[m] and bond_index[i][1] in brics_substructure_list[m]:
                    bond_contribution[i] = brics_substructure_contribution[m]
    brics_atom_contribution = deepcopy(atom_contribution)
    for m in range(len(brics_substructure_list)):
            for i in range(len(atom_list)):
                if atom_list[i][0] in brics_substructure_list[m]:
                    brics_atom_contribution[i] = brics_substructure_contribution[m]


    print('----结果已输出----')
    return atom_contribution, atom_list, brics_atom_contribution, bond_contribution, brics_substructure_contribution, brics_substructure_list


def ESOL_return_contribution(mol): 
    #真实标签(标准化)
    # y_true = ESOL_y_label[mol]
    y_true = ESOL_return_prediction(mol)
    #得到所有的BRICS片段序列
    mask_index = get_mask_index(ESOL_smile_canonical_remove_salt[mol])
    #得到所有键的index
    bond_index = get_bond_index_list(ESOL_smile_canonical_remove_salt[mol]) 
    #得到分子所有的原子序号列表
    mo = Chem.MolFromSmiles(ESOL_smile_canonical_remove_salt[mol])
    mol_list = range(mo.GetNumAtoms())
    #键的贡献度计算
    bond_contribution = []
    for m in range(len(bond_index)): #遍历所有的键
        #smile_list = re_sort(bond_index[m], ESOL_smile_canonical_remove_salt[mol])
        remove_bond_list = remove_mask_index(mol_list, bond_index[m])
        smile_list = Chem.MolFragmentToSmiles(mo, atomsToUse=remove_bond_list)
        #得到掩盖掉键并且数字化的list
        smile_num_list = [vocab[smile_list[k]] for k in range(len(smile_list))]
        #加入开始的0和结尾的0
        smile_num_list = _pad_start_end_token(smile_num_list)
        #将数字化之后的list填充到smiles_index的array中
        smiles_index = np.zeros((max_len + 2), dtype='int32')
        smiles_index[0:len(smile_num_list)] = np.array(smile_num_list)
    
        x_train = pd.DataFrame(np.hstack((smiles_index))).T
        y_pred = model_1.predict(x_train).tolist()[0][0]
        for model in [model_2, model_3, model_4, model_5]:
            y_pred += model.predict(x_train).tolist()[0][0]
        y_pred /= 5
        bond_contribution.append(y_true-y_pred) 
    #BRICS片段贡献度计算
    atom_list = []
    brics_bond_list = []
    brics_substructure_list = []
    atom_contribution = []
    brics_bond_contribution = []
    brics_substructure_contribution = []
    for m in range(len(mask_index)): #遍历所有的BRICS片段
        #smile_list = re_sort(mask_index[m], ESOL_smile_canonical_remove_salt[mol])
        remove_bond_list = remove_mask_index(mol_list, mask_index[m])
        if remove_bond_list == []:
            smile_list = Chem.MolToSmiles(mo)
        else:
            smile_list = Chem.MolFragmentToSmiles(mo, atomsToUse=remove_bond_list)
        #得到掩盖掉BRICS片段并且数字化的list
        smile_num_list = [vocab[smile_list[k]] for k in range(len(smile_list))]
        #加入开始的0和结尾的0
        smile_num_list = _pad_start_end_token(smile_num_list)
        #将数字化之后的list填充到smiles_index的array中
        smiles_index = np.zeros((max_len + 2), dtype='int32')
        smiles_index[0:len(smile_num_list)] = np.array(smile_num_list)
        x_train = pd.DataFrame(np.hstack((smiles_index))).T
        y_pred = model_1.predict(x_train).tolist()[0][0]
        for model in [model_2, model_3, model_4, model_5]:
            y_pred += model.predict(x_train).tolist()[0][0]
        y_pred /= 5
        if len(mask_index[m]) == 1:
            atom_contribution.append(y_true-y_pred)
            atom_list.append(mask_index[m])
        elif len(mask_index[m]) == 2:
            brics_bond_contribution.append(y_true-y_pred)
            brics_bond_list.append(sorted(mask_index[m]))
        else:
            brics_substructure_contribution.append(y_true-y_pred)
            mask_index_sorted = sorted(mask_index[m])
            brics_substructure_list.append(mask_index_sorted)

    for m in range(len(brics_substructure_list)):
            for i in range(len(bond_index)):
                if bond_index[i][0] in brics_substructure_list[m] and bond_index[i][1] in brics_substructure_list[m]:
                    bond_contribution[i] = brics_substructure_contribution[m]
    brics_atom_contribution = deepcopy(atom_contribution)
    for m in range(len(brics_substructure_list)):
            for i in range(len(atom_list)):
                if atom_list[i][0] in brics_substructure_list[m]:
                    brics_atom_contribution[i] = brics_substructure_contribution[m]

    print('----结果已输出----')
    return atom_contribution, atom_list, brics_atom_contribution, bond_contribution, brics_substructure_contribution, brics_substructure_list