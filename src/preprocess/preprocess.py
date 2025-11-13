import pandas
import requests
import tqdm
import json
import torch
from os import path
from rdkit.Chem import PandasTools, rdMolDescriptors, Descriptors
import argparse
from info import descriptors_to_keep
from rdkit import RDLogger
import pubchempy as pcp
import time
RDLogger.DisableLog("rdApp.*")
import cirpy

arg_parser = argparse.ArgumentParser(
    description="""
    Gather protein sequences, structures and checks its validity.
    Builds molecular descriptors and Morgan fingerprints for substrates.
    Remove any protein-substrate redundancies with HXKm dataset.
    """
)
arg_parser.add_argument(
    "--hxkm",
    help="Path to HXKm database.",
    default="../../data/hxkm.csv"
)
a = arg_parser.parse_args()

def build_molecular_features(df):
    # list ligands and smiles columns to build molecular features:
    ligands = df.loc[:, ["substrate", "smiles"]].drop_duplicates()
    ligands = ligands.dropna()

    # build RDKit mol object and remove all entries without mol object:
    PandasTools.AddMoleculeColumnToFrame(ligands, smilesCol="smiles", molCol="mol_obj")
    ligands = ligands.dropna(subset=["mol_obj"])
    ligands.reset_index(inplace=True, drop=True)

    # build descriptors and fingerprints:
    desc_list = [Descriptors.CalcMolDescriptors(mol) for mol in ligands["mol_obj"]]
    fingerprint_list = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in ligands["mol_obj"]]

    # mol object not needed anymore
    ligands.drop(["mol_obj"], axis=1, inplace=True)

    # make a dataframe for each of them:
    desc_df = pandas.DataFrame(desc_list)
    desc_df = desc_df.dropna(axis=1) # clean NaN columns
    desc_df = desc_df.loc[:, descriptors_to_keep]
    fingerprint_df = pandas.DataFrame([list(fp) for fp in fingerprint_list])

    print("Number of descriptors:", desc_df.shape[1])
    print("Number of fingerprint bits:", fingerprint_df.shape[1])

    # merge features and ligand identity:
    return pandas.concat([ligands, desc_df, fingerprint_df], axis=1)

def alphafold_check(uniprot_id, threshold=70):
    filepath = f"../../data/structures/{uniprot_id}.pdb"
    if not path.exists(filepath):
        return True
    with open(filepath) as file:
        atoms = [l for l in file if l[0:6].strip() == "ATOM" and l[12:16].strip() == "CA"]
    # b_factor contains the LDDT score of AF3:
    b_factors = torch.tensor([float(l[60:66].strip()) for l in atoms])
    # return true if 70% of pLDDT values are below a score of 70
    return True if (b_factors >= threshold).float().mean()*100 <= threshold else False

def fetch_sequence(uniprot_id):
    # if already fetched retrieved cached sequence:
    saved = "../../data/uniprotid_sequence.txt"
    with open(saved, "r") as f:
        lines = [l.replace("\n","").split(",") for l in f]
    uniprot_sequence = {l[0]: l[1] for l in lines}
    if uniprot_id in uniprot_sequence.keys():
        sequence = uniprot_sequence[uniprot_id]
        return None if sequence == "" else sequence
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Extract sequence from FASTA format
        lines = response.text.splitlines()
        sequence = ''.join(lines[1:].replace("\n", ""))  # Skip the first line which is the header
        with open(saved, "a") as f:
            f.write(",".join([uniprot_id, sequence]) + "\n")
        return sequence
    else:
        return None

def fetch_pdb(uniprot_id):
    out_file = f"../../data/structures/{uniprot_id}.pdb"
    if path.exists(out_file):
        return True
    metadata_res = requests.get(f"https://alphafold.com/api/prediction/{uniprot_id}")
    if metadata_res.status_code == 200:
        metadata = json.loads(metadata_res.content)[0]
        pdb_url = metadata["pdbUrl"]
    else:
        return False

    response = requests.get(pdb_url)
    if response.status_code == 200:
        with open(out_file, "wb") as f:
            f.write(response.content)
        return True
    return False

def fetch_smiles(name):
    saved = "../../data/substrate_smiles.txt"
    with open(saved, "r") as f:
        lines = [l.replace("\n","").split("\t") for l in f]
    substrate_smiles = {l[0]:l[1].replace("\n", "") for l in lines}
    if name in substrate_smiles.keys():
        smiles = substrate_smiles[name]
    else:
        try :
            smiles = cirpy.resolve(name, "smiles")
            smiles = smiles if smiles is not None else "None"
        except Exception as e:
            print(e)
            smiles = "None"
        with open(saved, "a") as f:
            f.write("\t".join([name, smiles]) + "\n")
    return smiles if smiles != "None" else None

# load data:
df = pandas.read_csv("../../data/brenda_sabio.csv")
hxkm = pandas.read_csv(a.hxkm)
hxkm.rename(columns={
    "Protein sequence": "sequence",
    "Substrate name": "substrate",
    "Substrate SMILES": "smiles",
    "Uniport ID": "uniprot_key",
    "KM value": "km_value",
    "Enzyme type": "protein_type",
    "EC Number": "enzyme_commission",
}, inplace=True)

df["sequence"] = None
df["below_threshold"] = None
df["smiles"] = None

hxkm["below_threshold"] = None

print("Fetching sequences and alphafold structures for brenda_sabio..")
uniprotid_sequence_path = "../../data/uniprotid_sequence.txt"
if not path.exists(uniprotid_sequence_path):
    open(uniprotid_sequence_path, "w")
# work only with entries which are not yet processed:
for key in tqdm.tqdm(df.uniprot_key.unique()):
    sequence = fetch_sequence(key)
    fetch_pdb(key)
    structure_check = alphafold_check(key)
    df.loc[df.uniprot_key==key, [
        "sequence", 
        "below_threshold"
    ]]= sequence, structure_check
print("brenda_sabio before droping empty sequence:", df.shape[0])
df.dropna(subset="sequence", inplace=True)
print("brenda_sabio after droping empty sequence:", df.shape[0])
df.to_csv("../../data/brenda_sabio.csv", index=False)
perc_above = (df.below_threshold == False).mean()*100
print(f"brenda_sabio.csv: 70% of pLDDT scores above 70: {perc_above:.2f}")

print("Fetching alphafold structures for hxkm..")
for key in tqdm.tqdm(hxkm.uniprot_key.unique()):
    fetch_pdb(key)
    structure_check = alphafold_check(key)
    hxkm.loc[hxkm.uniprot_key==key, ["below_threshold"]] = structure_check
hxkm.to_csv("../../data/hxkm.csv", index=False)
perc_above = (hxkm.below_threshold == False).mean()*100
print(f"hxkm.csv: 70% of pLDDT scores above 70: {perc_above:.2f}")

# fetch smiles:
print("Fetching SMILES..")
substrate_smiles_path = "../../data/substrate_smiles.txt"
if not path.exists(substrate_smiles_path):
    open(substrate_smiles_path, "w")
for substrate in tqdm.tqdm(df.substrate.unique()):
    if ";" in substrate:
        continue
    smiles = fetch_smiles(substrate)
    df.loc[df.substrate==substrate, ["smiles"]] = smiles
df.to_csv("../../data/brenda_sabio.csv", index=False)
perc_smiles = (~df.smiles.isna()).mean()*100
print(f"brends_sabio.csv: {perc_smiles:.2f} have a smiles.")

# filter brenda_sabio from entries that are found in hxkm.
# As described in 10.1186/s12859-024-05746-1 on `Independant Dataset Collection`,
# remove similar sequence-smiles from the data used for train.

print("brenda_sabio before filtering HXKm:", df.shape[0])
df_complex = df.substrate + df.sequence
hxkm_complex = hxkm.substrate + hxkm.sequence
df = df.loc[~df_complex.isin(hxkm_complex)]
print("brenda_sabio after filtering HXKm:", df.shape[0])

# merge features with the database:
df = pandas.merge(df, build_molecular_features(df), on=["substrate", "smiles"], how="inner")
hxkm = pandas.merge(hxkm, build_molecular_features(hxkm), on=["substrate", "smiles"], how="inner")
# save full df:
df.to_csv("../../data/brenda_sabio_processed.csv", index=False)
hxkm.to_csv("../../data/hxkm_processed.csv", index=False)
# reorder columns, for reproducibility reasons:
column_order = ["km_value", "uniprot_key", "below_threshold", "sequence", "smiles"]
remove_columns = [
    "substrate", "protein_type", "enzyme_commission", # brenda_sabio
    "Organism",# hxkm
    "Enzyme info",
    "Unit",
    "Reference"
]
df_columns = df.columns[
    (~df.columns.isin(column_order)) &
    (~df.columns.isin(remove_columns))
].tolist()
hxkm_columns = hxkm.columns[
    (~hxkm.columns.isin(column_order)) &
    (~hxkm.columns.isin(remove_columns))
].tolist()

df = df.loc[:, column_order + df_columns]
hxkm = hxkm.loc[:, column_order + hxkm_columns]
df.to_csv("../../data/csv/train_dataset_hxkm_complex.csv", index=False)
hxkm.to_csv("../../data/csv/HXKm_dataset_final_new.csv", index=False)
print("Finished gathering molecule and protein information.")


