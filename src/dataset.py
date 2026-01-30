from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import RobustScaler, MinMaxScaler, PowerTransformer
import numpy as np
import pandas as pd
import numpy

class KmClass(Dataset):
    def __init__(
            self, 
            df, 
            amino_scaler=None, 
            descriptor_scaler_robust=None, 
            descriptor_scaler_minmax=None, 
            km_scaler=None,
            with_seqid=True,
            test_mode=False,
            with_esm = False,
            esm_hf = None,
            only_as_esm = False,
            only_enz_esm = False,
            only_as = False
        ):
        self.dataframe = pd.DataFrame(df)
        self.test_mod = test_mode

        # handle protein feature size:
        self.with_seqid = with_seqid
        self.n_res_feats = (2,3)[with_seqid] # number of features per residue, depends on either with or without seqid feature
        self.n_prot_feats = 1024*self.n_res_feats # number of protein features
        self.tot_feats = self.n_prot_feats + 196 + 2048 # useful to provide as the input size for the model

        if "Ipc" in self.dataframe.columns:
            self.dataframe = self.dataframe.drop('Ipc', axis=1)
        self.dataframe = self.dataframe[self.dataframe['below_threshold'] == False] # drop datapoints with low alphafold confidence score
        self.dataframe = self.dataframe.loc[self.dataframe.km_value > 0]
        # clip km values to biological range
        self.dataframe['km_value'] = self.dataframe['km_value'].clip(0.00001, 1000)
        self.dataframe.sequence = self.dataframe.sequence.str.replace("\n", "")
        self.dataframe.dropna(subset=["sequence"], inplace=True)
        if "smiles" in self.dataframe.columns:
            self.dataframe.drop(columns=["smiles"], inplace=True)
        self.dataframe['conditioned_bs'] = self.dataframe['conditioned_bs'].apply(
            lambda x: [int(i) for i in x.strip("[]\"'").split(',')] if isinstance(x, str) else x)
        self.dataframe.drop(self.dataframe[
            self.dataframe['conditioned_bs'].isna()
        ].index, inplace=True)
        print("Before drop:", self.dataframe.shape[0])
        self.dataframe.drop(self.dataframe[
            self.dataframe["sequence"].str.len() != self.dataframe["conditioned_bs"].str.len()
        ].index, inplace=True)
        print("After drop:", self.dataframe.shape[0])

        # Use pre-fitted scalers if provided
        self.amino_scaler = amino_scaler
        self.descriptor_scaler_robust = descriptor_scaler_robust
        self.descriptor_scaler_minmax = descriptor_scaler_minmax
        self.km_scaler = km_scaler

        # amino acid identities
        aa_1LC = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        aa_encoded = torch.tensor([ord(aa) for aa in aa_1LC]).unsqueeze(dim=1).numpy()

        # Fit the scaler on amino acids if it's not provided
        if self.amino_scaler is None:
            self.amino_scaler = MinMaxScaler(feature_range=(0, 1))
            self.amino_scaler.fit(aa_encoded)

        # Fit the scalers on the descriptors if not provided
        descriptors = self.dataframe.iloc[:, 4:-2049].values  # Descriptor columns only
        if self.descriptor_scaler_robust is None:
            self.descriptor_scaler_robust = PowerTransformer()
            descriptors_scaled_robust = self.descriptor_scaler_robust.fit_transform(descriptors)
        else:
            descriptors_scaled_robust = self.descriptor_scaler_robust.transform(descriptors)
        
        # Fit MinMaxScaler on the robust-scaled descriptors if not provided
        if self.descriptor_scaler_minmax is None:
            self.descriptor_scaler_minmax = MinMaxScaler(feature_range=(0, 1))
            self.descriptor_scaler_minmax.fit(descriptors_scaled_robust)

        # When fitting
        descriptors = self.dataframe.iloc[:, 4:-2049].values  # Descriptor columns only
        print(f"Loaded Km class. Size of the database: {len(self):,}")
        print("Number of descriptor features when fitting:", descriptors.shape[1])

        # Fit the scaler on the km values if it's not provided
        if self.km_scaler is None:
            self.km_scaler = MinMaxScaler(feature_range=(0, 1))
            # new scaling: set the range for km values between 10^-7 and 10^-1 M
            self.km_log_capped = np.log10(self.dataframe['km_value']).values.reshape(-1, 1)
            #km_values_capped = np.where(self.valid_data['km_value'] > 100, 100, np.where(self.valid_data['km_value'] < 0.0001, 0.0001, self.valid_data['km_value']))
            #self.km_log_capped = np.log10(km_values_capped).reshape(-1, 1)
            self.km_scaler.fit(self.km_log_capped)
        
        self.with_esm = with_esm
        if with_esm:
            # self.seq_esm = pickle.load(open("../data/esm_embeddings.pkl", "rb"))
            self.esm_hf = esm_hf
            self.only_as_esm = only_as_esm
            self.only_ezm_esm = only_enz_esm
            esm_sequences = list(self.esm_hf.keys())
            print("Before filtering out sequences without esm:", self.dataframe.shape[0])
            self.dataframe = self.dataframe.loc[self.dataframe.sequence.isin(esm_sequences)]
            print("After filtering out sequences without esm:", self.dataframe.shape[0])
            self.tot_feats = (4804, 3524)[only_as_esm or only_enz_esm]

        self.only_as = only_as
        if only_as:
            self.n_res_feats = 2 # number of features per residue, depends on either with or without seqid feature
            self.n_prot_feats = 1024*self.n_res_feats # number of protein features
            self.tot_feats = self.n_prot_feats + 196 + 2048 # useful to provide as the input size for the model

    def __len__(self):
        return len(self.dataframe)
    
    def __get_pocket_values__(self,row):
        # Retrieve the pocket values from the 'conditioned_bs' column
        pocket_values = row['conditioned_bs']
        # Check if pocket_values is a list, string, or numpy array
        if isinstance(pocket_values, list):
            return pocket_values
        elif isinstance(pocket_values, str):
            import ast
            return ast.literal_eval(pocket_values)
        else:
            return list(pocket_values)
    
    def __structure_data__(self, sequence, pocket_values, row):
        # Create a list for alternating sequence and pocket values
        combined = []
        
        amino_acid_values = [ord(aa) for aa in sequence]  # Convert amino acids to ASCII values

        # Scale the amino acid ASCII values
        if not self.only_as:
            amino_acid_values_scaled = self.amino_scaler.transform(np.array(amino_acid_values).reshape(-1, 1)).flatten()

        # Combine scaled amino acid values with pocket values
        for i, pocket in enumerate(pocket_values):
            if not self.only_as:
                combined.append(amino_acid_values_scaled[i])  # Get the scaled value from the list
            combined.append(pocket)
            if self.with_seqid:
                combined.append(i/1024)

        if len(combined) < self.n_prot_feats: # each residue encoded with 3 features
            combined.extend([0] * (self.n_prot_feats - len(combined)))  # Fill with zeros
        else:
            combined = combined[:self.n_prot_feats]  # Truncate if it exceeds
        
        # Apply both scalers to the descriptors
        descriptors = row.iloc[4:-2049].values.reshape(1, -1)
        descriptors_scaled_robust = self.descriptor_scaler_robust.transform(descriptors)
        descriptors_scaled = self.descriptor_scaler_minmax.transform(descriptors_scaled_robust)

        combined.extend(descriptors_scaled.flatten())  # Combine scaled descriptors
        
        # Add fingerprint information
        fingerprints = row.iloc[-2049:-1].values.reshape(1, -1)

        combined.extend(fingerprints.flatten())
        
        return combined
    
    def esm_substrate_input(self, idx):
        row = self.dataframe.iloc[idx]

        # Extract the data
        protein_id = row['uniprot_key']
        sequence = row['sequence']
        esm_embeddings = torch.tensor(numpy.array(self.esm_hf[sequence])[0])
        sequence = pd.Series(list(sequence))
        km_value = row['km_value']

        # Ensure km_value is not NaN
        if pd.isna(km_value):
            print(f"Skipping {protein_id}: km_value is NaN")
            return None 
        
        # fetch and build pocket info:
        pocket_values = self.__get_pocket_values__(row)
        active_site_indices = torch.tensor(pocket_values).nonzero().flatten()

        # build active site esm-representation:
        active_site_esm = esm_embeddings[active_site_indices+1].mean(dim=0) # sequence tokens starts at position 1

        # fetch sequence esm embedding:
        sequence_esm = esm_embeddings.mean(dim=0)
        
        # concatenate sequence + active site embeddings, depending on which ESM to keep:
        if self.only_as_esm:
            enzyme_features = active_site_esm.tolist()
        elif self.only_ezm_esm:
            enzyme_features = sequence_esm.tolist()
        else:
            enzyme_features = torch.cat([sequence_esm, active_site_esm]).tolist()

        # Apply both scalers to the descriptors
        descriptors = row.iloc[4:-2049].values.reshape(1, -1)
        descriptors_scaled_robust = self.descriptor_scaler_robust.transform(descriptors)
        descriptors_scaled = self.descriptor_scaler_minmax.transform(descriptors_scaled_robust).flatten().tolist()
        
        # Add fingerprint information
        fingerprints = row.iloc[-2049:-1].values.tolist()

        input_features = [*enzyme_features, *descriptors_scaled, *fingerprints] # n_features: 4,804
        
        return input_features

    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Extract the data
        protein_id = row['uniprot_key']
        sequence = row['sequence']
        km_value = row['km_value']

        # Ensure km_value is not NaN
        if pd.isna(km_value):
            print(f"Skipping {protein_id}: km_value is NaN")
            return None 
        
        # Retrieve pocket values
        pocket_values = self.__get_pocket_values__(row)

        # If pocket values are None, skip this protein (no binding site prediction available)
        if pocket_values is None or len(pocket_values) != len(sequence):
            print("None pocket or len pocket is not len sequence")
            return None

        # get rid of unwanted pocket values (higher than 1)
        for i, x in enumerate(pocket_values):
            if x != 1: pocket_values[i] = 0
            # pocket_values[i] = 0

        if self.with_esm:
            combined_data = self.esm_substrate_input(idx)
        else:
            combined_data = self.__structure_data__(sequence, pocket_values, row)
        
        x = torch.tensor(combined_data, dtype=torch.float32)
        y = torch.tensor([km_value], dtype=torch.float32).log10()
        y = torch.tensor(self.km_scaler.transform(y.unsqueeze(1)),dtype=torch.float32).flatten()

        # Check for NaNs in `x`
        if torch.isnan(x).any() or torch.isnan(y).any():
            print("None in x or y")
            return None

        return (x, y) if self.test_mod == False else (x,y,torch.tensor([idx]))

if __name__ == "__main__":
    import argparse
    import h5py

    arg_parser = argparse.ArgumentParser(
        description="Contains the class to load the Km dataset."
    )
    arg_parser.add_argument(
        "--csv",
        help="Path to the pre-processed BRENDA and Sabio csv containing features.",
        default="../data/csv/train_dataset_hxkm_complex_conditioned_bs.csv"
    )
    a = arg_parser.parse_args()

    df = pd.read_csv(a.csv)
    esm_hf = h5py.File("../data/esm_embeddings.hdf5", "r")
    ds = KmClass(
        df, 
        with_esm=False, 
        esm_hf=esm_hf, 
        only_as_esm=False, 
        only_enz_esm=False,
        only_as=True
    )
    print("number of features:", ds.tot_feats)
    x,y = ds[0]
    print("input shape:", x.shape)
    # sequence = ds.dataframe.iloc[0].sequence
    # seq_len = len(sequence)
    # print("total n features:", ds.tot_feats)
    # print("y:", y)
    # print("x shape:", x.shape)
    # print("fingerprint:", x[-2049:])
    # print("fingerprint unique values:", x[-2049:].unique())
    # print("bs:", x[1:seq_len*ds.n_res_feats:ds.n_res_feats])
    # print("bs unique values:", x[1:seq_len*ds.n_res_feats:ds.n_res_feats].unique())
    # print("descriptors:", x[ds.n_prot_feats:ds.n_prot_feats+196])
    # print("descriptors unique values:", x[ds.n_prot_feats:ds.n_prot_feats+196].unique())
    # print("aa id:", x[0:seq_len*ds.n_res_feats:ds.n_res_feats])
    # print("aa id unique values:", x[0:seq_len*ds.n_res_feats:ds.n_res_feats].unique())
    # if ds.with_seqid:
    #     print("aa position:", x[2:seq_len*ds.n_res_feats:ds.n_res_feats])
    #     print("aa position unique values:", x[2:seq_len*ds.n_res_feats:ds.n_res_feats].unique())
    # print("len sequence:", seq_len)
    # print("sequence:", sequence)