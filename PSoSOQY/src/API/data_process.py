from substructure_generate import get_mask_index, get_fg_index
from rdkit.Chem import BRICS
import pandas as pd
import re
import numpy as np
from data_generate import vocab
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from data_preprocess import remove_salt_stereo
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from rdkit.Chem import BRICS
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
REMOVER = SaltRemover()
regex_pattern=r'te|se|Cl|Br|Na|Te|Se|[./\\#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'
max_len=100


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


#光敏分子模型
# model = keras.models.load_model("D:\桌面\project\deep learning\mask_BiLSTM\model\BiLSTM_attention.h5", compile = False)
model_1 = keras.models.load_model("D:\桌面\project\deep learning\mask_BiLSTM\model\BiLSTM_attention.h5", compile = False)
model_2 = keras.models.load_model("D:\桌面\project\deep learning\mask_BiLSTM\model\BiLSTM_attention_2.h5", compile = False)
model_3 = keras.models.load_model("D:\桌面\project\deep learning\mask_BiLSTM\model\BiLSTM_attention_3.h5", compile = False)
model_4 = keras.models.load_model("D:\桌面\project\deep learning\mask_BiLSTM\model\BiLSTM_attention_4.h5", compile = False)
model_5 = keras.models.load_model("D:\桌面\project\deep learning\mask_BiLSTM\model\BiLSTM_attention_5.h5", compile = False)
model_6 = keras.models.load_model("D:\桌面\project\deep learning\mask_BiLSTM\model\BiLSTM_attention_6.h5", compile = False)
model_7 = keras.models.load_model("D:\桌面\project\deep learning\mask_BiLSTM\model\BiLSTM_attention_7.h5", compile = False)
model_8 = keras.models.load_model("D:\桌面\project\deep learning\mask_BiLSTM\model\BiLSTM_attention_8.h5", compile = False)
model_9 = keras.models.load_model("D:\桌面\project\deep learning\mask_BiLSTM\model\BiLSTM_attention_9.h5", compile = False)
model_10 = keras.models.load_model("D:\桌面\project\deep learning\mask_BiLSTM\model\BiLSTM_attention_10.h5", compile = False)

print("-----模型加载完成-----")

#将mol对象转化为svg的图像
def mol_to_svg(mol):
    d2d = rdMolDraw2D.MolDraw2DSVG(600,600)
    #d2d.drawOptions().addAtomIndices=True
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = SVG(d2d.GetDrawingText())
    return svg  

#去掉需mask的子结构index
def remove_mask_index(list1, list2):
    #原始列表
    original_list = list1
    #要删除的元素列表
    elements_to_remove = list2
    #使用列表推导式创建一个新列表，只包含不在要删除的元素列表中的原始列表元素
    filtered_list = [x for x in original_list if x not in elements_to_remove]
    
    return filtered_list



def return_prediction(PS, solvent, ref):
    #将分子SMILES以及溶剂SMILES、参比物SMILES分成一个一个字符
    PS_canonical = canonical_smiles(PS)
    PS_canonical_remove_salt = remove_salt_stereo(PS_canonical, REMOVER)
    solvent_canonical = canonical_smiles(solvent)
    solvent_canonical_remove_salt = remove_salt_stereo(solvent_canonical, REMOVER)
    ref_canonical = canonical_smiles(ref)
    ref_canonical_remove_salt = remove_salt_stereo(ref_canonical, REMOVER)
    smile_list = re.findall(regex_pattern, PS_canonical_remove_salt)
    solvent_list = re.findall(regex_pattern, solvent_canonical_remove_salt)
    ref_list = re.findall(regex_pattern, ref_canonical_remove_salt)
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
    y_pred = model_1.predict(x_train)
    for model in [model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10]:
            y_pred += model.predict(x_train)
    y_pred /= 10
    return y_pred, PS

def return_contribution(PS, solvent, ref): 
    #将分子SMILES以及溶剂SMILES、参比物SMILES分成一个一个字符
    PS_canonical = canonical_smiles(PS)
    PS_canonical_remove_salt = remove_salt_stereo(PS_canonical, REMOVER)
    solvent_canonical = canonical_smiles(solvent)
    solvent_canonical_remove_salt = remove_salt_stereo(solvent_canonical, REMOVER)
    ref_canonical = canonical_smiles(ref)
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
    #真实标签
    y_true, PS_SMILES = return_prediction(PS, solvent, ref)
    #得到所有的BRICS片段序列
    mask_index, atom_symbol = get_mask_index(PS_canonical_remove_salt)
    #得到所有的官能团序列
    fg_index, fg_name = get_fg_index(PS_canonical_remove_salt)
    #得到分子所有的原子序号列表
    mol = Chem.MolFromSmiles(PS_canonical_remove_salt)
    mol_list = range(mol.GetNumAtoms())
    
    #保存需要传到前端的BRICS片段数据列表
    table_list1 = []
    #保存需要传到前端的官能团数据列表
    table_list2 = []
    #描述列表
    description_list = []
    #BRICS片段贡献度计算
    brics_substructure_list = []
    brics_substructure_contribution = []
    for m in range(len(mask_index)): #遍历所有的BRICS片段
        #前端数据列表里的字典对象
        keys = ['BRICS_substructure', 'SMILES', 'Index', 'Attribution']
        values = []

        #smile_list = re_sort(mask_index[m], smile_canonical_remove_salt[mol])
        remove_bond_list = remove_mask_index(mol_list, mask_index[m])
        smile_list = Chem.MolFragmentToSmiles(mol, atomsToUse=remove_bond_list)
        #得到掩盖掉BRICS片段并且数字化的list
        smile_num_list = [vocab[smile_list[k]] for k in range(len(smile_list))]
        #加入开始的0和结尾的0
        smile_num_list = _pad_start_end_token(smile_num_list)
        #将数字化之后的list填充到smiles_index的array中
        smiles_index = np.zeros((max_len + 2), dtype='int32')
        smiles_index[0:len(smile_num_list)] = np.array(smile_num_list)
        x_train = pd.DataFrame(np.hstack((smiles_index, solvent_index, ref_index))).T
        #取10个模型的平均值
        y_pred = model_1.predict(x_train)
        for model in [model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10]:
            y_pred += model.predict(x_train)
        y_pred /= 10
        if len(mask_index[m]) <= 2:
            continue
        else:
            m = Chem.MolFromSmiles(PS_canonical_remove_salt)
            mol_smiles = Chem.MolFragmentToSmiles(m, atomsToUse=mask_index[m])
            mol = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(m, atomsToUse=mask_index[m]))
            substructure_svg = mol_to_svg(mol)
            values.append(substructure_svg)
            values.append(mol_smiles)
            values.append(mask_index[m])
            values.append(y_true-y_pred.tolist()[0][0])
            table_list1.append(dict(zip(keys, values)))
            # brics_substructure_contribution.append(y_true-y_pred.tolist()[0][0])
            # mask_index_sorted = sorted(mask_index[m])
            # brics_substructure_list.append(mask_index_sorted)

    for m in range(len(fg_index)): #遍历所有的官能团片段
        #前端数据列表里的字典对象
        keys = ['Functional_group', 'Attribution']
        values = []
        number_fg = []
        for n in range(len(fg_index[m])):
            for k in fg_index[m][n]:
                number_fg.append(k)
        remove_bond_list = remove_mask_index(mol_list, number_fg)
        smile_list = Chem.MolFragmentToSmiles(mol, atomsToUse=remove_bond_list)
        #得到掩盖掉BRICS片段并且数字化的list
        smile_num_list = [vocab[smile_list[k]] for k in range(len(smile_list))]
        #加入开始的0和结尾的0
        smile_num_list = _pad_start_end_token(smile_num_list)
        #将数字化之后的list填充到smiles_index的array中
        smiles_index = np.zeros((max_len + 2), dtype='int32')
        smiles_index[0:len(smile_num_list)] = np.array(smile_num_list)
        x_train = pd.DataFrame(np.hstack((smiles_index, solvent_index, ref_index))).T
        #取10个模型的平均值
        y_pred = model_1.predict(x_train)
        for model in [model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10]:
            y_pred += model.predict(x_train)
        y_pred /= 10
        m = Chem.MolFromSmiles(PS_canonical_remove_salt)
        mol = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(m, atomsToUse=number_fg))
        substructure_svg = mol_to_svg(mol)
        values.append(substructure_svg)
        values.append(y_true-y_pred.tolist()[0][0])
        table_list2.append(dict(zip(keys, values)))

    print('----结果已输出----')
    return {
        'table1': table_list1,
        'table2': table_list2,
        'SMILES': PS_SMILES,
        'Predicted_value': y_true
    }


# def return_info(PS, solvent, ref):
#     #将分子SMILES以及溶剂SMILES、参比物SMILES分成一个一个字符
#     PS_canonical = canonical_smiles(PS)
#     PS_canonical_remove_salt = remove_salt_stereo(PS_canonical, REMOVER)
#     solvent_canonical = canonical_smiles(solvent)
#     solvent_canonical_remove_salt = remove_salt_stereo(solvent_canonical, REMOVER)
#     ref_canonical = canonical_smiles(ref)
#     ref_canonical_remove_salt = remove_salt_stereo(ref_canonical, REMOVER)
#     solvent_list = re.findall(regex_pattern, solvent_canonical_remove_salt)
#     ref_list = re.findall(regex_pattern, ref_canonical_remove_salt)

#     #分别创建溶剂，参比物的array矩阵，长度为max_len+2
#     solvent_index = np.zeros((max_len + 2), dtype='int32')
#     ref_index = np.zeros((max_len + 2), dtype='int32')

#     #将溶剂与参比物的SMILES编码为数字
#     solvent_num_list = [vocab[solvent_list[k]] for k in range(len(solvent_list))]
#     solvent_num_list = _pad_start_end_token(solvent_num_list)
#     ref_num_list = [vocab[ref_list[k]] for k in range(len(ref_list))]
#     ref_num_list = _pad_start_end_token(ref_num_list)

#     solvent_index[0:len(solvent_num_list)] = np.array(solvent_num_list)
#     ref_index[0:len(ref_num_list)] = np.array(ref_num_list)
#     #真实标签
#     y_true, PS_SMILES = return_prediction(PS, solvent, ref)
#     return y_true, PS_SMILES