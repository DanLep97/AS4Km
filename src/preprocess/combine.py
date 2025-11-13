import pandas

# load SABIO-RK
sabio = pandas.read_csv("../../data/sabio.csv")

# change the column names for compatibility:
sabio.rename(columns={
    "UniprotID": "uniprot_key",
    "ECNumber": "enzyme_commission",
    "Substrate": "substrate",
    "Value": "km_value", 
    "EnzymeType": "protein_type"
}, inplace=True)

# wildtype is WT:
print("Sabio unique protein types: ", sabio.protein_type.unique())
sabio.protein_type = sabio.protein_type.str.replace("wildtype", "WT")
print("Sabio unique protein types: ", sabio.protein_type.unique())
sabio = sabio.loc[:, [
    "uniprot_key",
    "enzyme_commission",
    "substrate",
    "km_value",
    "protein_type"
]]

# load brenda:
# Km values are in mM in Brenda. See `km_value` field in https://www.brenda-enzymes.org/schemas/2.0.0/enzyme.schema.json
# Therefore, no need to change anything.

brenda = pandas.read_csv("../../data/brenda.csv")

# concat both:
brenda_sabio = pandas.concat([brenda, sabio])
brenda_sabio.to_csv("../../data/brenda_sabio_with_mutants.csv", index=False)

# making sure only wildtypes:
brenda_sabio = brenda_sabio.loc[brenda_sabio.protein_type == "WT"]
brenda_sabio.to_csv("../../data/brenda_sabio.csv", index=False)
print(f"brenda_sabio.csv file saved with {brenda_sabio.shape[0]} entries.")