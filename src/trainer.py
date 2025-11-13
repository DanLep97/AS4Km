import argparse
from dataset import KmClass
from model import Network, trainer
from sklearn.model_selection import KFold, train_test_split
import pandas
import torch
from torch import nn, optim
import os
from torch.utils.data import DataLoader
from hyperparameters import hyperparameters
import pickle
from copy import deepcopy

RANDOM_SEED=123
torch.manual_seed(RANDOM_SEED)

arg_parser = argparse.ArgumentParser(
    description="""
    Train the model using K fold crossvalidation and save optimized weights in ../data/models/<name of model>/<k>_folds.pth.
"""
)
arg_parser.add_argument(
    "--name",
    help="Name of the trained model.",
    default="test"
)
arg_parser.add_argument(
    "--folds",
    help="Number K of folds to perform.",
    default=0,
    type=int
)
arg_parser.add_argument(
    "--runs",
    help="Number of folds to use. If runs == folds, all folds are used.",
    default=2,
    type=int
)
arg_parser.add_argument(
    "--epochs",
    help="Number of epochs per folds.",
    default=2,
    type=int
)
arg_parser.add_argument(
    "--db",
    help="Database to split k times.",
    default="../data/csv/train_dataset_hxkm_complex_conditioned_bs.csv"
)
arg_parser.add_argument(
    "--k",
    help="Specifies the k-th fold being loaded. Only to use when --folds=0."
)
arg_parser.add_argument(
    "--with-seqid",
    help="Adds the residue position in the sequence as a residue-level feature.",
    default=False,
    action="store_true"
)
arg_parser.add_argument(
    "--gated",
    help="Includes the gated mechanism in the first layer of the FFCNN.",
    default=False,
    action="store_true"
)

a = arg_parser.parse_args()

# initialize model, optimizer:
device = ("cpu", "cuda")[torch.cuda.is_available()]
print("loaded device:", device)

# create the folder to save the folds:
# today_str = datetime.datetime.now().strftime("%d%m%Y")
folder_name = f"../data/models/{a.name}"
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

# initialize Kfolds
df = pandas.read_csv(a.db)
df_indices = torch.arange(df.shape[0])

if a.folds > 0:
    folder = KFold(
        n_splits=a.folds,
        shuffle=True, # at each initialization the order of indices is different 
        random_state=RANDOM_SEED
    )
    folds = folder.split(df_indices)
    for k, (train_indices, validation_indices) in enumerate(folds):
        if k >= a.runs:
            break
        print(f"Starting training fold {k}")
        train_df = df.iloc[train_indices]
        train_df.reset_index(inplace=True, drop=True) # reset the index and drop the column of the old index

        validation_df = df.iloc[validation_indices]
        validation_df.reset_index(inplace=True, drop=True) # reset the index and drop the column of the old index

        train_data = KmClass(train_df, with_seqid=a.with_seqid)
        scalers = {
            "amino_scaler": train_data.amino_scaler,
            "descriptor_scaler_robust": train_data.descriptor_scaler_robust,
            "descriptor_scaler_minmax": train_data.descriptor_scaler_minmax,
            "km_scaler": train_data.km_scaler,
            "a": a # save the arguments of the run as well
        }
        pickle.dump(scalers, open(f"{folder_name}/{k+1}_fold_scalers.pkl", "wb"))

        val_data = KmClass(
            validation_df,
            amino_scaler=scalers["amino_scaler"],
            descriptor_scaler_robust=scalers["descriptor_scaler_robust"],
            descriptor_scaler_minmax=scalers["descriptor_scaler_minmax"],
            km_scaler=scalers["km_scaler"], 
            with_seqid=a.with_seqid
        )
        net = Network(
            hidden_dim1=hyperparameters["hidden_dim1"], 
            hidden_dim2=hyperparameters["hidden_dim2"], 
            hidden_dim3=hyperparameters["hidden_dim3"], 
            dropout1=hyperparameters["dropout1"], 
            dropout2=hyperparameters["dropout2"],
            with_gate=a.gated,
            input_size=train_data.tot_feats
        ).to(device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            net.parameters(), 
            lr=hyperparameters["learning_rate"], 
            weight_decay=hyperparameters["weight_decay"]
        ) 
        trainer(
            epochs=a.epochs, 
            train_dataset=train_data, 
            val_dataset=val_data, 
            best_model_path=f"{folder_name}/{k+1}_fold_model.pth",
            metrics_path=f"{folder_name}/{k+1}_fold_metrics.pth",
            criterion=criterion,
            optimizer=optimizer,
            net=net
        )
else:
    k = a.k
    train_indices, validation_indices = train_test_split(df_indices, test_size=.25)

    train_df = df.iloc[train_indices]
    train_df.reset_index(inplace=True, drop=True) # reset the index and drop the column of the old index

    validation_df = df.iloc[validation_indices]
    validation_df.reset_index(inplace=True, drop=True) # reset the index and drop the column of the old index

    train_data = KmClass(train_df)
    scalers = {
        "amino_scaler": deepcopy(train_data.amino_scaler),
        "descriptor_scaler_robust": deepcopy(train_data.descriptor_scaler_robust),
        "descriptor_scaler_minmax": deepcopy(train_data.descriptor_scaler_minmax),
        "km_scaler": deepcopy(train_data.km_scaler)
    }
    pickle.dump(scalers, open(f"{folder_name}/{k}_fold_scalers.pkl", "wb"))

    val_data = KmClass(
        validation_df,
        amino_scaler=scalers["amino_scaler"],
        descriptor_scaler_robust=scalers["descriptor_scaler_robust"],
        descriptor_scaler_minmax=scalers["descriptor_scaler_minmax"],
        km_scaler=scalers["km_scaler"] 
    )

    net = Network(
        hidden_dim1=hyperparameters["hidden_dim1"], 
        hidden_dim2=hyperparameters["hidden_dim2"], 
        hidden_dim3=hyperparameters["hidden_dim3"], 
        dropout1=hyperparameters["dropout1"], 
        dropout2=hyperparameters["dropout2"],
        with_gate=a.gated,
        input_size=train_data.tot_feats
    ).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        net.parameters(), 
        lr=hyperparameters["learning_rate"], 
        weight_decay=hyperparameters["weight_decay"]
    ) 
    trainer(
        epochs=a.epochs, 
        train_dataset=train_data, 
        val_dataset=val_data, 
        best_model_path=f"{folder_name}/{k}_fold_model.pth",
        metrics_path=f"{folder_name}/{k}_fold_metrics.pth",
        criterion=criterion,
        optimizer=optimizer,
        net=net
    )
