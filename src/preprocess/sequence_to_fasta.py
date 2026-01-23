import pandas

brenda_sabio_df = pandas.read_csv("../../data/brenda_sabio.csv")
hxkm_df = pandas.read_csv("../../data/hxkm.csv")
# prepare fasta files for genetic clustering:
train_fastas = ""
test_fastas = ""

for i in range(brenda_sabio_df.shape[0]):
    case = brenda_sabio_df.iloc[i]
    fasta_str = f">{case.uniprot_key}\n{case.sequence}\n"
    train_fastas += fasta_str
for i in range(hxkm_df.shape[0]):
    case = hxkm_df.iloc[i]
    fasta_str = f">{case.uniprot_key}\n{case.sequence}\n"
    test_fastas += fasta_str

with open("../../data/train_fastas.fasta", "w") as train_f:
    train_f.write(train_fastas)
with open("../../data/test_fastas.fasta", "w") as test_f:
    test_f.write(test_fastas)
