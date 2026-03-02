"""
kNN-median baseline for Km prediction using fingerprint similarity.

This script:
1) loads train and test CSVs,
2) applies dataset-like filtering (e.g., drop invalid rows, ensure sequence/mask length match),
3) extracts the last N fingerprint columns (default: 2048),
4) predicts each test label as the median log10(Km) of the top-K most similar training samples
   based on a Tanimoto-like similarity on fingerprint vectors,
5) reports metrics (R2/MSE/RMSE/Pearson) and optionally saves a similarity histogram plot.

Example usage
-------------
python knn_baseline.py \
  --train-path data/csv/train_dataset_hxkm_complex_conditioned_bs.csv \ 
  --test-path data/csv/HXKm_dataset_final_new_unconditioned_bs.csv \ 
  --out-dir data/csv/knn_baseline \
  --k 10 \
  --n-fp 2048
"""

import argparse
import os
import random
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# matplotlib is optional if user disables plotting
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute a kNN-median baseline for log10(Km) using fingerprint similarity."
    )
    p.add_argument("--train-path", type=str, required=True,
                   help="Path to training CSV.")
    p.add_argument("--test-path", type=str, required=True,
                   help="Path to test CSV.")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Directory to write outputs (metrics + optional plots).")

    p.add_argument("--k", type=int, default=10,
                   help="Number of nearest neighbors used for the median (default: 10).")
    p.add_argument("--n-fp", type=int, default=2048,
                   help="Number of fingerprint columns taken from the END of the table (default: 2048).")

    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--no-plot", action="store_true",
                   help="Disable saving the similarity histogram plot.")
    p.add_argument("--hist-test-index", type=int, default=0,
                   help="Which test row index to use for similarity histogram (default: 0).")
    return p


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def apply_dataset_like_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply filtering to mimic the preprocessing expected by the model/dataset:
    - drop Ipc and smiles if present
    - remove below_threshold rows, km_value <= 0
    - clip km_value
    - ensure sequence exists
    - parse conditioned_bs if present and enforce length match
    """
    df = df.copy()

    if "Ipc" in df.columns:
        df = df.drop(columns=["Ipc"])

    # match your previous behavior (KmClass-like)
    if "below_threshold" in df.columns:
        df = df.loc[df["below_threshold"] == False].copy()  # noqa: E712

    if "km_value" not in df.columns:
        raise ValueError("Expected column 'km_value' not found.")

    df = df.loc[df["km_value"] > 0].copy()
    df["km_value"] = df["km_value"].clip(0.00001, 1000)

    if "sequence" not in df.columns:
        raise ValueError("Expected column 'sequence' not found.")
    df["sequence"] = df["sequence"].astype(str).str.replace("\n", "", regex=False)
    df = df.dropna(subset=["sequence"])

    if "smiles" in df.columns:
        df = df.drop(columns=["smiles"])

    # If conditioned_bs exists, parse and enforce length match (as in your script)
    if "conditioned_bs" in df.columns:
        def _parse_mask(x):
            if isinstance(x, str):
                x = x.strip()
                # handle strings like "[0, 1, 0]" (optionally quoted)
                x = x.strip("\"'")  # remove surrounding quotes if present
                x = x.strip("[]")
                if x.strip() == "":
                    return None
                return [int(i) for i in x.split(",")]
            return x

        df["conditioned_bs"] = df["conditioned_bs"].apply(_parse_mask)
        df = df.dropna(subset=["conditioned_bs"])
        df = df.loc[df["sequence"].str.len() == df["conditioned_bs"].str.len()].copy()

        # downstream baseline uses fingerprints only, so we drop it here
        df = df.drop(columns=["conditioned_bs"])

    return df


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Tanimoto-like similarity for non-binary fingerprints using min/max generalization.
    Works for binary as well.
    """
    intersection = np.sum(np.minimum(fp1, fp2))
    union = np.sum(np.maximum(fp1, fp2))
    return 0.0 if union == 0 else float(intersection / union)


def extract_fingerprints(df: pd.DataFrame, n_fp: int) -> np.ndarray:
    if df.shape[1] < n_fp:
        raise ValueError(f"Dataframe has {df.shape[1]} columns, cannot take last {n_fp} as fingerprints.")
    X = df.iloc[:, -n_fp:].values
    if X.shape[1] != n_fp:
        raise RuntimeError("Fingerprint extraction failed unexpectedly.")
    return X


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, Optional[float]]:
    r2 = float(r2_score(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    if HAS_SCIPY:
        r, p = pearsonr(y_true, y_pred)
        return r2, mse, rmse, float(r), float(p)
    else:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
        return r2, mse, rmse, r, None


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_data = pd.read_csv(args.train_path)
    test_data = pd.read_csv(args.test_path)

    train_data = apply_dataset_like_filters(train_data)
    test_data = apply_dataset_like_filters(test_data)

    print("Filtered train:", train_data.shape)
    print("Filtered test:", test_data.shape)

    X_train = extract_fingerprints(train_data, args.n_fp)
    X_test = extract_fingerprints(test_data, args.n_fp)

    y_train_log = np.log10(train_data["km_value"].values)
    y_test_log = np.log10(test_data["km_value"].values)

    # similarity histogram for one test example (optional)
    if not args.no_plot:
        j = args.hist_test_index
        if j < 0 or j >= X_test.shape[0]:
            raise ValueError(f"--hist-test-index {j} out of range for test set size {X_test.shape[0]}")
        sims = np.array([tanimoto_similarity(X_test[j], X_train[i]) for i in range(X_train.shape[0])])
        print(f"Similarity stats (test[{j}] vs train): min={sims.min():.4f} max={sims.max():.4f} mean={sims.mean():.4f}")

        plt.hist(sims, bins=50)
        plt.xlabel("Tanimoto Similarity")
        plt.ylabel("Frequency")
        plt.title(f"Tanimoto Similarity Distribution: test[{j}] vs training")
        fig_path = os.path.join(args.out_dir, "tanimoto_similarity_distribution.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved histogram: {fig_path}")

    # kNN-median baseline
    K = int(args.k)
    if K <= 0:
        raise ValueError("--k must be >= 1")
    global_fallback = float(np.median(y_train_log))

    y_pred_test = []
    for j in range(X_test.shape[0]):
        sims = np.array([tanimoto_similarity(X_test[j], X_train[i]) for i in range(X_train.shape[0])])

        if np.max(sims) == 0:
            y_pred = global_fallback
        else:
            topk_idx = np.argsort(sims)[-K:]
            y_pred = float(np.median(y_train_log[topk_idx]))

        y_pred_test.append(y_pred)

    y_pred_test = np.array(y_pred_test)

    r2, mse, rmse, r, p = compute_metrics(y_test_log, y_pred_test)

    print("Overall R2 on test (log10 KM):", r2)
    print("Overall MSE on test (log10 KM):", mse)
    print("Overall RMSE on test (log10 KM):", rmse)
    print("Pearson r:", r, ("p-value: " + str(p) if p is not None else "(scipy not installed; p-value N/A)"))

    # save metrics to file
    metrics_path = os.path.join(args.out_dir, "knn_baseline_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"train_path: {args.train_path}\n")
        f.write(f"test_path: {args.test_path}\n")
        f.write(f"k: {K}\n")
        f.write(f"n_fp: {args.n_fp}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"R2: {r2}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"Pearson_r: {r}\n")
        f.write(f"Pearson_p: {p if p is not None else 'N/A'}\n")

    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
