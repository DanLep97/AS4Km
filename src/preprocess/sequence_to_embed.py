import pandas
import tqdm
import torch
from transformers import EsmTokenizer, EsmModel
import h5py
from os import path
from rdkit import DataStructs
import numpy

# first, embed enzymes:
model_name="facebook/esm2_t33_650M_UR50D"
brenda_sabio_df = pandas.read_csv("../../data/brenda_sabio.csv")
hxkm_df = pandas.read_csv("../../data/hxkm.csv")
sequences = set([*brenda_sabio_df.sequence.tolist(), *hxkm_df.sequence.tolist()])

# load model and tokenizer:
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval() # Set to evaluation mode

with h5py.File("../../data/esm_embeddings.hdf5", "a") as hf:
    # process sequences that don't have an embedding:
    embedded_seqs = list(hf.keys())
    sequences = [seq for seq in sequences if seq not in embedded_seqs]
    print(f"Processing {len(sequences):,} sequences...")
    for sequence in tqdm.tqdm(sequences):
        if len(sequence) > 1024:
            continue
        try:
            inputs = tokenizer(sequence, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state.cpu()
            hf.create_dataset(sequence, data=last_hidden_states)
        except Exception as e:
            print(f"error for sequence {sequence}")
            print(e)
            continue

# second, prepare fasta files for genetic clustering:
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
