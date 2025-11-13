import pandas

def convert_unit(value, unit):
    """
    Given the unit, converts the value into mM.
    """
    if unit == 'M' :
        value = value * 1000
        unit = 'mM'
    if unit == 'M^2' :
        value = value ** (1/2) * 1000
        unit = 'mM'
    return value, unit

input_tsv = "../../data/KM_sabio_4_unisubstrate.tsv"
db = pandas.read_csv(input_tsv, sep="\t")

# convert values to mM
db[["Value", "Unit"]] = db.apply(
    lambda row: convert_unit(row.Value, row.Unit), 
    axis=1, result_type="expand"
)

# keep only wildtypes and valid values:
print(db.shape)
db = db.loc[
    (db.EnzymeType == "wildtype") & # only wildtypes
    (db.Unit == "mM") & # only with mM unit
    (db.Type == "Km") & # only Km parameters (sanity check)
    (~db.Value.isna()) # keep out empty Km measurements
]
db.to_csv("../../data/sabio.csv", index=False)
print(f"Saved sabio.csv with {db.shape[0]} entries.")