import matplotlib.cm as cm
import matplotlib
from IPython.display import SVG, display
import matplotlib.colors as mcolors
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Geometry
rdDepictor.SetPreferCoordGen(True)


sns.set(color_codes=True)

def atom_attribution_visualize(smiles, atom_attribution, number = -1, cmap_name='seismic_r',):
    mol = Chem.MolFromSmiles(smiles)
    cmap = cm.get_cmap(cmap_name, 10)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    highlight_atom_colors = {}
    atomrads = {}

    for i in range(mol.GetNumAtoms()):
        highlight_atom_colors[i] = [plt_colors.to_rgba(float(atom_attribution[i]))]
        atomrads[i] = 0.2
    rdDepictor.Compute2DCoords(mol)

    # now draw the molecule, with highlights:
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
    dos = drawer.drawOptions()
    dos.useBWAtomPalette()
    drawer.DrawMoleculeWithHighlights(mol, smiles, highlight_atom_colors, {}, atomrads, {})
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    drawer.WriteDrawingText('./deep learning/mask_BiLSTM/data/BRICS_contribution/mol_{}_atom.png'.format(number))
    return png


def sub_attribution_visualize(smiles, atom_attribution, 
                              bond_attribution, 
                              ring_attribution, atom_list=None, 
                              bond_list=None,
                              ring_list=None, number = -1, cmap_name='seismic_r', sub_type='brics'):
    mol = Chem.MolFromSmiles(smiles)
    cmap = cm.get_cmap(cmap_name, 10)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1) 
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    highlight_atom_colors = {}
    highlight_bond_colors = {}
    atomrads = {}
    widthmults = {}
    if atom_list is None:
        atom_list = range(0, mol.GetNumAtoms())
    if bond_list is None:
        bond_list = range(0, mol.GetNumBonds())
    for i in atom_list:
        highlight_atom_colors[i] = [plt_colors.to_rgba(float(atom_attribution[atom_list.index(i)]))]
        atomrads[atom_list.index(i)] = 0.2
    if len(bond_list) > 0:
        for i in bond_list:
            highlight_bond_colors[i] = [plt_colors.to_rgba(float(bond_attribution[bond_list.index(i)]))]
            widthmults[bond_list.index(i)] = 1
    if len(ring_list)>0:
        ring_color = [plt_colors.to_rgba(float(ring_attribution[i])) for i in range(len(ring_list))]
    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
    dos = drawer.drawOptions()
    dos.useBWAtomPalette()
#     drawer.SetFontSize(1)

    if len(ring_list)>0:
        # a hack to set the molecule scale
        drawer.DrawMoleculeWithHighlights(mol, smiles, highlight_atom_colors, highlight_bond_colors, atomrads, widthmults)
        drawer.ClearDrawing()
        conf = mol.GetConformer()
        for i in range(len(ring_list)):
            aring = ring_list[i]
            ring_colors_i = ring_color[i]
            ps = []
            for aidx in aring:
                pos = Geometry.Point2D(conf.GetAtomPosition(aidx))
                ps.append(pos)
            drawer.SetFillPolys(True)
            drawer.SetColour(ring_colors_i)
            drawer.DrawPolygon(ps)
        dos.clearBackground = False

    # now draw the molecule, with highlights:
    drawer.DrawMoleculeWithHighlights(mol, smiles, highlight_atom_colors, highlight_bond_colors, atomrads, widthmults)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    drawer.WriteDrawingText('./data/BRICS_contribution/mol_{}_{}.png'.format(number, sub_type))
    return png






