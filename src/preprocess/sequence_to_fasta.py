import pandas

brenda_sabio_df = pandas.read_csv("../../data/brenda_sabio.csv")
hxkm_df = pandas.read_csv("../../data/hxkm.csv")
# prepare fasta files for genetic clustering:
train_fastas = ""
test_fastas = ""

brenda_sabio_enz_sequence = {
    enz: brenda_sabio_df.loc[brenda_sabio_df.uniprot_key == enz].sequence.values[0]
    for enz in brenda_sabio_df.uniprot_key.unique()
}
hxkm_enz_sequence = {
    enz: hxkm_df.loc[hxkm_df.uniprot_key == enz].sequence.values[0]
    for enz in hxkm_df.uniprot_key.unique()
}
for enz, seq in brenda_sabio_enz_sequence.items():
    train_fastas += f">{enz}\n{seq}\n"

for enz, seq in hxkm_enz_sequence.items():
    test_fastas += f">{enz}\n{seq}\n"

with open("../../data/train_fastas.fasta", "w") as train_f:
    train_f.write(train_fastas)
with open("../../data/test_fastas.fasta", "w") as test_f:
    test_f.write(test_fastas)
