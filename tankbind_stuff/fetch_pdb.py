import argparse
import glob

arg_parser = argparse.ArgumentParser(
    description="""
Fetch the pdbs using either the pdb code if available or the alphafold prediction.
"""
)
arg_parser.add_argument(
    "--pdb",
    help="Path to where all pdb are stored.",
    default="../data/protein_pdb"
)
a = arg_parser.parse_args()

pdbs = glob.glob(f"{a.pdb}/*.pdb")
pdbs_uniprot_ids = [p.split("")]
print("Number of pdbs:", len(pdbs))