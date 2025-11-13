import argparse
from dataset import KmClass
from model import Network, tester
import pandas
from hyperparameters import hyperparameters
import torch
from torch.nn import MSELoss
from torcheval.metrics.functional import r2_score
from torchmetrics import PearsonCorrCoef
import glob
import os
import pickle

arg_parser = argparse.ArgumentParser(
    description="""
    Loads a trained model and uses it for inference on the database provided.
"""
)
arg_parser.add_argument(
    "--db",
    help="Path to the test csv file.",
)
arg_parser.add_argument(
    "--model",
    help="Name of the model to test."
)
arg_parser.add_argument(
    "--csv-output",
    help="Save the output of the test here. It includes the r2, pearson correlation score.",
    default="../data/csv/inferences.csv"
)
arg_parser.add_argument(
    "--name",
    help="Name of the run to save in the csv file."
)
arg_parser.add_argument(
    "--k",
    help="Use only when there is no need to loop through folds.",
    default=0,
    type=int
)
a = arg_parser.parse_args()

# load data
df = pandas.read_csv(a.db)
device = ("cpu", "cuda")[torch.cuda.is_available()]

# load model:
folder = f"../data/models/{a.model}"
all_r2 = []
all_pcs = []
all_loss = []

# metrics:
pcc = PearsonCorrCoef()
criterion = MSELoss()

if a.k == 0:
    model_files = glob.glob(f"{folder}/*_fold_model.pth")
    for model_file in model_files:
        print(f"Testing {model_file}...")
        k_th = model_file.split("/")[-1].split("_")[0]
        scalers = pickle.load(open(f"{folder}/{k_th}_fold_scalers.pkl", "rb"))

        # run inference on test:
        test_ds = KmClass(
            df,
            amino_scaler=scalers["amino_scaler"],
            descriptor_scaler_robust=scalers["descriptor_scaler_robust"],
            descriptor_scaler_minmax=scalers["descriptor_scaler_minmax"],
            km_scaler=scalers["km_scaler"], 
            with_seqid=scalers["a"].with_seqid,
            test_mode=True
        )

        net = Network(
            hidden_dim1=hyperparameters["hidden_dim1"], 
            hidden_dim2=hyperparameters["hidden_dim2"], 
            hidden_dim3=hyperparameters["hidden_dim3"], 
            dropout1=hyperparameters["dropout1"], 
            dropout2=hyperparameters["dropout2"],
            with_gate=scalers["a"].gated,
            input_size=test_ds.tot_feats
        ).to(device)
        params = torch.load(model_file)["model_state_dict"]
        net.load_state_dict(params)

        # run inference on test
        all_preds, all_y, all_idx = tester(test_ds, net)
        outputs = {
            "y_unscaled": scalers["km_scaler"].inverse_transform(all_y.view(-1,1)),
            "preds_unscaled": scalers["km_scaler"].inverse_transform(all_preds.view(-1,1)),
            "y_scaled": all_y.view(-1,1),
            "preds_scaled": all_preds.view(-1,1),
            "all_idx": all_idx
        }
        pickle.dump(outputs, open(f"{folder}/{k_th}_fold_{a.name}_outputs.pkl", "wb"))
        r2 = r2_score(all_preds, all_y)
        pcs = pcc(all_preds, all_y)
        loss = criterion(all_preds, all_y)

        all_r2.append(r2.item())
        all_pcs.append(pcs.item())
        all_loss.append(loss.item())

        print("R2:", r2)
        print("Pearson Correlation Coefficient score:,", pcs)

    # save metrics:
    n = len(model_files)
    metrics = pandas.DataFrame({
        "name": [a.name]*n,
        "fold": list(range(n)),
        "model": [a.model]*n,
        "csv_test": [a.db]*n,
        "n": [len(test_ds)]*n,
        "r2": all_r2,
        "pearson": all_pcs,
        "mse": all_loss,
    })
else:
    # load model weights and scalers:
    df = pandas.read_csv(a.db)
    scalers = pickle.load(open(f"{folder}/{a.k}_fold_scalers.pkl", "rb"))
    test_ds = KmClass(
        df,
        amino_scaler=scalers["amino_scaler"],
        descriptor_scaler_robust=scalers["descriptor_scaler_robust"],
        descriptor_scaler_minmax=scalers["descriptor_scaler_minmax"],
        km_scaler=scalers["km_scaler"] 
    )
    params = torch.load(f"{folder}/{a.k}_fold_model.pth")["model_state_dict"]
    net = Network(
        hidden_dim1=hyperparameters["hidden_dim1"], 
        hidden_dim2=hyperparameters["hidden_dim2"], 
        hidden_dim3=hyperparameters["hidden_dim3"], 
        dropout1=hyperparameters["dropout1"], 
        dropout2=hyperparameters["dropout2"]
    ).to(device)
    net.load_state_dict(params)

    # run inference on test
    all_preds, all_y = tester(test_ds, net)
    outputs = {
        "y_unscaled": scalers["km_scaler"].inverse_transform(all_y.view(-1,1)),
        "preds_unscaled": scalers["km_scaler"].inverse_transform(all_preds.view(-1,1)),
        "y_scaled": all_y.view(-1,1),
        "preds_scaled": all_preds.view(-1,1)
    }
    pickle.dump(outputs, open(f"{folder}/{a.k}_fold_{a.name}_outputs.pkl", "wb"))

    r2 = r2_score(all_preds, all_y)
    pcs = pcc(all_preds, all_y)
    loss = criterion(all_preds, all_y)

    all_r2.append(r2.item())
    all_pcs.append(pcs.item())
    all_loss.append(loss.item())

    print("R2:", r2)
    print("Pearson Correlation Coefficient score:,", pcs)
    # save metrics:
    metrics = pandas.DataFrame({
        "name": [a.name],
        "fold": [a.k],
        "model": [a.model],
        "csv_test": [a.db],
        "n": [len(test_ds)],
        "r2": all_r2,
        "pearson": all_pcs,
        "mse": all_loss,
    })

if not os.path.exists(a.csv_output):
    metrics.to_csv(a.csv_output, index=False)
else:
    df = pandas.read_csv(a.csv_output)
    df = pandas.concat([df, metrics], axis=0)
    df.to_csv(a.csv_output, index=False)
