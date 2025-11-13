import argparse
import pandas
import os
import tqdm
import torch
from ast import literal_eval
import numpy

arg_parser = argparse.ArgumentParser(
    description="""
    Requires as input the <condtioned_bs> file obtained using the tankbind_stuff/build_bs.py script

    Use this script to modify the binding vector information to the `conditioned_bs` feature for csv databases
    without conditioned binding sites. This is for compatibility between different types of databases.
    Typically, this is used for the ablation study where we investigate the impact of binding site feature not conditioned on the substrate.

    Also, builds different features in seperate csv files.
    """
)
arg_parser.add_argument(
    "--db",
    help="The database to which `conditioned_bs` column feature is added."
)
arg_parser.add_argument(
    "--p2rank-out",
    help="Folder where the p2rank <uniprot_key>_residues.csv file is found.",
    default="../../data/p2rank"
)
a = arg_parser.parse_args()

unconditioned_file = a.db.replace("conditioned", "unconditioned")
fingerprint_free_file = a.db.replace("conditioned_bs", "fingerprint_free")
description_free_file = a.db.replace("conditioned_bs", "descriptors_free")
bs_free_file = a.db.replace("conditioned_bs", "bs_free")
aa_id_free_file = a.db.replace("conditioned_bs", "aa_id_free")
protein_free_file = a.db.replace("conditioned_bs", "protein_free")
molecule_free_file = a.db.replace("conditioned_bs", "molecule_free")

# build the unconditioned binding-site feature
df = pandas.read_csv(a.db)
binding_sites = []
print("Building unconditioned binding sites...")
for i in tqdm.tqdm(range(df.shape[0])):
    row = df.iloc[i]
    p2rank_file = f"{a.p2rank_out}/{row.uniprot_key}.pdb_residues.csv"
    if os.path.exists(p2rank_file):
        pocket = pandas.read_csv(f"{a.p2rank_out}/{row.uniprot_key}.pdb_residues.csv")[" pocket"].values
        pocket[pocket != 1] = 0 # all cases where it's not 0 is 0
        binding_sites.append(pocket.tolist())
    else:
        binding_sites.append(None)
df["conditioned_bs"] = binding_sites
df.to_csv(unconditioned_file, index=False)
print("Unconditioned file save at:", unconditioned_file)

# build the descriptor-free csv:
df = pandas.read_csv(a.db)
df.drop(columns=["smiles"], inplace=True)
df.drop('Ipc', axis=1, inplace=True)
df.iloc[:, 4:-2049] = 0
df.to_csv(description_free_file, index=False)
print("Description free file save at:", description_free_file)

# build the fingerprint-free csv:
df = pandas.read_csv(a.db)
df.drop(columns=["smiles"], inplace=True)
df.drop('Ipc', axis=1, inplace=True)
df.iloc[:, -2049:-1] = 0
df.to_csv(fingerprint_free_file, index=False)
print("Fingerprint free file save at:", fingerprint_free_file)

# build the binding-site free csv:
df = pandas.read_csv(a.db)
df.sequence = df.sequence.str.replace("\n", "")
df.conditioned_bs = df.sequence.apply(lambda x: [0]*len(x))
df.to_csv(bs_free_file, index=False)
print("Binding site free file save at:", bs_free_file)

# identity free csv:
df = pandas.read_csv(a.db)
aas = []
print("building identity free csv")
for i in tqdm.tqdm(range(df.shape[0])):
    row = df.iloc[i]
    # to remove the amino acid identity, the sequence is only made of alanines:
    aa = pandas.Series(["K" for i in range(len(row.sequence))])
    if not pandas.isna(row.conditioned_bs):
        pocket = literal_eval(row.conditioned_bs)
        if len(pocket) == len(row.sequence):
            bs_mask = torch.tensor(pocket, dtype=torch.bool)
            bs_mask_index = bs_mask.nonzero().flatten().tolist()
            sequence = pandas.Series(list(row.sequence))
            # keep the aa identity of only the binding site:
            aa[bs_mask_index] = sequence[bs_mask_index]
    aas.append("".join(aa.tolist()))
df.sequence = aas
df.to_csv(aa_id_free_file, index=False)
print("Amino acid identity free file saved at:", aa_id_free_file)

# protein free csv:
df = pandas.read_csv(a.db)
df.sequence = df.sequence.apply(lambda x: "".join(["A"]*len(x)))
df.conditioned_bs = df.sequence.apply(lambda x: [0]*len(x))
df.to_csv(protein_free_file, index=False)
print("Protein free file saved at:", protein_free_file)

# molecule free csv:
df = pandas.read_csv(a.db)
df.drop(columns=["smiles"], inplace=True)
df.drop('Ipc', axis=1, inplace=True)
df.iloc[:, 4:-2049] = 0
df.iloc[:, -2049:-1] = 0
df.to_csv(molecule_free_file, index=False)
print("Molecule free file saved at:", molecule_free_file)