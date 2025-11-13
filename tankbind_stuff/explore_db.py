import argparse
import pandas

arg_parser = argparse.ArgumentParser(
    description="""
    File used to explore the output of tankbind and compare to the original db.
"""
)
arg_parser.add_argument(
    "--ranked-pockets",
    help="The output of the `build_binding_site_vectors.py` file.",
    default="../data/tankbind/ranked_pockets.csv"
)
arg_parser.add_argument(
    "--db",
    help="Path to the database csv.",
    default="../data/csv/final_df_fingerprints_alphafold_hxkm_complex.csv"
)
a = arg_parser.parse_args()

rp = pandas.read_csv(a.ranked_pockets)
db = pandas.read_csv(a.db)

# print unique protein-ligand complexes:
u_rp = rp.groupby(["protein_name", "compound_name"]).size()
u_db = db.groupby(["uniprot_key", "smiles"]).size()
print(u_rp)
print(u_db)
