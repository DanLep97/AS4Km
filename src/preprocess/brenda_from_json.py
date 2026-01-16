import json
import pickle
import pandas
import argparse
import re

arg_parser = argparse.ArgumentParser(
    description="""
    Generate the .csv file with organized and processed data from the raw .json database.
    The json file can be fetched and stored localy from: https://www.brenda-enzymes.org/download.php
    """
)
arg_parser.add_argument(
    "--json-file",
    help="Path to the .json database. Default: ../../databases/brenda_2023_1.json",
    default="../data/brenda_2023_1.json"
)
a = arg_parser.parse_args()

db_path = a.json_file
# description of each fields:
# https://www.brenda-enzymes.org/schemas/docs/1.0.0/brenda.schema.html#data_pattern1_synonyms


with open(db_path) as f:
    print("Loading data...")
    db_data = json.load(f)["data"]

ecs = list(db_data.keys())[1:] # EC (enzyme commission) numbers

ecs_proteins = {}
print("Populating ecs_proteins..")
for ec in ecs: #populate protein list
    if "proteins" in db_data[ec]:
        for prot in db_data[ec]["proteins"].keys():
            ecs_proteins[f"{ec}_{prot}"] = db_data[ec]["proteins"][prot]

#targets:
uniprot_keys = [] # KM values with only 1 protein
substrates = [] # The substrate from the KM value having 1 protein
km_values = [] # Corresponding KM value
protein_types = [] # Recombinant, wild type
enzyme_commissions = []

#filtered out entries:
ec_with_no_protein_km = []

print("Acquiring targets..")
for ec in ecs: # get the values:
    if "km_value" in db_data[ec] and "protein" in db_data[ec]:
        for km_creds in db_data[ec]["km_value"]:
            if "value" in km_creds and "value" in km_creds and "proteins" in km_creds and len(km_creds["proteins"]) == 1:
                protein_num = km_creds["proteins"][0]
                if "accessions" in db_data[ec]["protein"][protein_num]:
                    if "comment" not in km_creds:
                        continue
                    else:
                        protein_type = ("WT", "recomb_or_mutant")[re.search(r"recomb|mutant", km_creds["comment"]) != None]
                        protein_types.append(protein_type)
                        substrates.append(km_creds["value"])
                        km_values.append(km_creds["value"])
                        uniprot_key = db_data[ec]["protein"][protein_num]["accessions"][0]
                        uniprot_keys.append(uniprot_key)
                        enzyme_commissions.append(ec)
    else:
        ec_with_no_protein_km.append(ec)

print("Saving into csv..")
data = {
    "uniprot_key": uniprot_keys,
    "substrate": substrates,
    "km_value": km_values,
    "protein_type": protein_types,
    "enzyme_commission": enzyme_commissions
}
df = pandas.DataFrame(data)
print("BRENDA database including WT and mutants:", df.shape[0])
df = df.loc[
    df.protein_type == "WT"
]
print("BRENDA database including WT only:", df.shape[0])
df.to_csv("../../data/brenda.csv", index=False)
print(f"brenda.csv saved with {df.shape[0]} entries.")