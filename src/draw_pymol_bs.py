import pandas
df = pandas.read_csv("../data/tankbind/ranked_pockets.csv")
ranked_pockets = df.loc[
    df.groupby([
        'protein_name', 'compound_name'
    ],sort=False)['affinity'].agg('idxmax')
].reset_index()
target_enzyme = "P20228"

residue_file = pandas.read_csv(f"../data/p2rank/{target_enzyme}.pdb_residues.csv")
all_pocket = residue_file[" pocket"].values

conditioned_pocket = int(ranked_pockets.loc[
    ranked_pockets.protein_name == target_enzyme
].pocket_name.item().replace("pocket_", ""))
top_ranked_pocket = 1
conditioned_res = "resi " + "+".join(residue_file[" residue_label"][
    all_pocket == conditioned_pocket
].astype(str).tolist())
top_ranked_res = "resi " +"+".join(residue_file[" residue_label"][
    all_pocket == top_ranked_pocket
].astype(str).tolist())
print("Conditioned residues:", conditioned_res)
print("Top ranked residues:", top_ranked_res)
# top1_residues = 
# binding_sites.append(pocket.tolist())