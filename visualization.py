import pandas as pd
import numpy as np
from utils import atom_attribution_visualize, sub_attribution_visualize
from IPython.display import Image
from Substructure_contribution import return_contribution, ESOL_return_contribution, return_prediction
from Substructure_contribution import smile_canonical_remove_salt, ESOL_smile_canonical_remove_salt
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
mol = 106

atom_contribution, atom_list, brics_atom_contribution, brics_bond_contribution, brics_substructure_contribution, brics_substructure_list = return_contribution(mol)
#atom_contribution, bond_contribution, substructure_contribution, substructure_list = return_attribution(mol)
# atom_contribution, atom_list, brics_atom_contribution, brics_bond_contribution, brics_substructure_contribution, brics_substructure_list = ESOL_return_contribution(mol)

y_pred = return_prediction(mol)

smile = smile_canonical_remove_salt[mol]
#smile = 'CN1/C(=C/C=C/C=C/C=C/C2=[N+](C)c3ccc(F)cc3C2(C)C)C(C)(C)c2cc(F)ccc21'

# pd.DataFrame(atom_contribution).to_csv('./deep learning/mask_BiLSTM/data/BRICS_contribution/atom_contribution_{}.csv'.format(mol))
# pd.DataFrame(brics_substructure_contribution).to_csv('./deep learning/mask_BiLSTM/data/BRICS_contribution/brics_substructure_contribution_{}.csv'.format(mol))
# pd.DataFrame(brics_substructure_list).to_csv('./deep learning/mask_BiLSTM/data/BRICS_contribution/brics_substructure_list_{}.csv'.format(mol))

# substructure_list = [[25, 26, 27, 26], 
#                      [0, 1, 2, 3, 4, 3, 5, 6, 7, 44, 45, 44, 37, 38, 39, 40, 41, 40, 39, 38, 37, 9, 8, 7, 8, 9,
#                       10, 11, 30, 28, 29, 28, 13, 12, 11, 12, 13, 14, 15, 24, 21, 23, 21, 22, 21, 17, 16, 15, 16, 17,
#                       18, 19, 20, 5, 3, 2, 1], 
#                      [31, 32, 33, 34, 33, 32], 
#                      #[21, 22, 23, 24, 25, 24, 26, 27, 26, 28, 29, 28, 30, 31, 32, 33, 34, 33, 35, 36, 35, 37, 38, 37, 39, 40]

# ]

print('The predict value of mol_{} is {}'.format(mol, y_pred))

print("********{}********".format('atom attribution'))
display(Image(atom_attribution_visualize(smile, atom_contribution, number=mol, cmap_name='RdBu_r')))

print("********{}********".format('brics attribution'))
print(atom_contribution)
display(Image(sub_attribution_visualize(smile, brics_atom_contribution, brics_bond_contribution, brics_substructure_contribution, ring_list= brics_substructure_list, number=mol, cmap_name='RdBu_r',sub_type='brics')))
print()


