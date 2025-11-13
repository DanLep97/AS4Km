import sys
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import pandas as pd
import os
import subprocess
import torch
import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')

arg_parser = argparse.ArgumentParser(
    description="""
    Build the binding site vector
"""
)
arg_parser.add_argument(
    "--tankbind",
    help="Path to tankbind package. Default: /home/daniil/Desktop/Ph.D/TankBind/tankbind",
    default="../../tankbind_stuff/TankBind/tankbind"
)
arg_parser.add_argument(
    "--tankbind-model",
    help="Path to self_dock.pt model.",
    default="../../tankbind_stuff/TankBind/saved_models/self_dock.pt"
)
arg_parser.add_argument(
    "--csv-input",
    help="Path to the input csv containing annotations and paths to pdb files.",
    default="../../data/csv/train_dataset_hxkm_complex.csv"
)
arg_parser.add_argument(
    "--pdb-folder",
    help="Path to receptor PDB files.",
    default="../../data/structures"
)
arg_parser.add_argument(
    "--p2rank-out",
    help="Path to the output of the p2rank binary. Default: ../../databases/TankBind/p2rank",
    default="../../data/p2rank"
)
arg_parser.add_argument(
    "--output-path",
    help="Path where the database for tankbind will be saved to be loaded from.",
    default="../../data/tankbind/"
)
a = arg_parser.parse_args()
sys.path.insert(0, a.tankbind)
# tankbind imports:
from feature_utils import get_protein_feature
from feature_utils import extract_torchdrug_feature_from_mol
from data import TankBind_prediction
from model import get_model
from generation_utils import get_LAS_distance_constraint_mask, get_info_pred_distance

# load the data
df = pd.read_csv(a.csv_input)
df.dropna(subset=["below_threshold"], inplace=True)
df = df.loc[df.below_threshold == False]
pocket_info_csv = f"{a.output_path}/ranked_pockets.csv"
# build protein features:
failed_receptors = 0
no_pdb_file = 0
print("Building protein features..")
protein_feats = {}
for uniprot_key in tqdm(df.uniprot_key.unique()):
    protein_file = f"{a.pdb_folder}/{uniprot_key}.pdb"
    if not os.path.exists(protein_file):
        no_pdb_file+=1
        continue
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(uniprot_key, protein_file)
    res_list = list(s.get_residues())
    try:
        protein_features = get_protein_feature(res_list)
    except Exception as e:
        failed_receptors+=1
        continue
    protein_feats[uniprot_key] = protein_features

# build compound features:
failed_ligands = 0
print("Building compound features..")
compound_feats = {}
for smiles in tqdm(df.smiles.unique()):
    mol = Chem.MolFromSmiles(smiles)
    mol.Compute2DCoords()
    try:
        compound_features = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
    except Exception as e:
        failed_ligands+=1
        continue
    compound_feats[smiles] = compound_features

print("Failed to read pdbs:", no_pdb_file)
print("Failed ligand feature generation:", failed_ligands)
print("Failed receptor feature generation", failed_receptors)

# filter df for cases with computed protein and compound features:
df_for_db = df.loc[(df.uniprot_key.isin(protein_feats.keys())) & (df.smiles.isin(compound_feats.keys()))]
df_for_db.reset_index(inplace=True, drop=True) # reset the indices, don't save originals (not needed)

# build database:
print("Building dataset...")
info = []
for (uniprot_key, smiles),_ in tqdm(df_for_db.groupby(["uniprot_key", "smiles"])):
    # get the protein's potential binding sites
    p2rankFile = f"{a.p2rank_out}/{uniprot_key}.pdb_predictions.csv"

    # save all pockets to be probed by the predictor:
    pocket = pd.read_csv(p2rankFile)
    pocket.columns = pocket.columns.str.strip()
    pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values
    for ith_pocket, com in enumerate(pocket_coms):
        com = ",".join([str(a.round(3)) for a in com])
        info.append([
            uniprot_key, smiles, f"pocket_{ith_pocket+1}", com
        ])
info = pd.DataFrame(info, columns=[
    'protein_name', 'compound_name', 'pocket_name', 'pocket_com'
])
print(info)

# build the torch dataset:
if os.path.exists(a.output_path):
    os.system(f"rm -rf {a.output_path}")
    os.system(f"mkdir {a.output_path}")
else:
    os.system(f"mkdir {a.output_path}")
dataset = TankBind_prediction(
    a.output_path, data=info, protein_dict=protein_feats, compound_dict=compound_feats
)
print("Dataset built.")

# load the model:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=1
print("device used: ", device)
logging.basicConfig(level=logging.INFO)
model = get_model(0, logging, device)
model.load_state_dict(torch.load(a.tankbind_model, map_location=device))
model.eval()
model = model.to(device)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    follow_batch=["x","y", "compound_pair"],
    shuffle=False,
    num_workers=batch_size,
)
y_pred_list = []
print(f"Running predictions with batch size {batch_size}..")
BAs = []
preds = []
for i, data in enumerate(tqdm(dataloader)):
    # try:
        data = data.to(device)
        with torch.no_grad():
            y_pred, affinity_pred = model(data)
        BAs.append(affinity_pred.detach().cpu())
        for i in range(data.y_batch.max() + 1):
            preds.append((y_pred[data['y_batch'] == i]).detach().cpu())
info["affinity"] = torch.cat(BAs)

# save info
info.to_csv(pocket_info_csv, index=False)

# group by protein, compound and select the best affinity as the conditioned binding site:
ranked_pockets = info.loc[info.groupby(['protein_name', 'compound_name'],sort=False)['affinity'].agg('idxmax')].reset_index()

# based on the best pockets, build the new binding site conditioned on the compound:
print("Building conditioned binding site vectors..")
bs_vectors = []
no_bs_vectors = 0
for i in tqdm(range(df.shape[0])):
    case = df.iloc[i]

    # this file contains info on which residues belongs to which pocket
    residue_file = f"{a.p2rank_out}/{case.uniprot_key}.pdb_residues.csv" 

    if not os.path.exists(residue_file):
        bs_vectors.append(None)
        continue

    residue_df = pd.read_csv(residue_file)
    residue_df.columns = residue_df.columns.str.strip() # fix formating

    pocket_rows = ranked_pockets.loc[
        (ranked_pockets.protein_name == case.uniprot_key) &
        (ranked_pockets.compound_name == case.smiles)
    ].pocket_name
    if pocket_rows.shape[0] != 1: # not all enzyme-substrate complex have a pocket
        bs_vectors.append(None)
        continue
    pocket = int(pocket_rows.item().split("_")[-1])
    bs_vector = torch.zeros(residue_df.shape[0], dtype=torch.long)
    bs_mask = (residue_df.pocket == pocket).tolist()
    bs_vector[bs_mask] = 1
    if (bs_vector == 1).nonzero().shape[0] == 0:
        no_bs_vectors+=1
        bs_vectors.append(None)
    else:
        bs_vectors.append(bs_vector.tolist())
df["conditioned_bs"] = bs_vectors
print("Number of residue.csv files without binding site vector:", no_bs_vectors)
conditioned_bs_file = f"{a.csv_input.replace('.csv', '')}_conditioned_bs.csv"
df.to_csv(conditioned_bs_file, index=False)
print(f"The database with conditioned files is saved at {conditioned_bs_file}.")
print(f"Length of the db: {df.shape[0]}, Number of cases without conditioned binding site: {df.loc[df.conditioned_bs.isna()].shape[0]}")

# in case we need 3D coordinates of the compound inside the most probable BS:
# print("Computing heavy atom coordinates for top 1 BA score for each protein-ligand complex..")
# outputs = {}
# for i in tqdm(ranked_ba["index"]):
#         name = dataset.data.iloc[i].protein_name
#         coords = dataset[i].coords
#         protein_nodes_xyz = dataset[i].node_xyz
#         n_compound = coords.shape[0]
#         n_protein = protein_nodes_xyz.shape[0]
#         y_pred = preds[i].reshape(n_protein, n_compound)
#         y = dataset[i].dis_map.reshape(n_protein, n_compound)
#         compound_pair_dis_constraint = torch.cdist(coords, coords)
#         pred_dist_info = get_info_pred_distance(coords, y_pred, protein_nodes_xyz,
#             compound_pair_dis_constraint,
#             n_repeat=1, show_progress=False
#         )
#         new_coords = pred_dist_info.sort_values("loss")['coords'].iloc[0]
#         outputs[name] = new_coords

# print(f"Computed {len(outputs)} complexes.")
# torch.save(outputs, f"{a.output_path}/outputs.pth")
