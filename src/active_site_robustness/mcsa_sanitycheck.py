"""
Compute overlap metrics between predicted binding-site masks (e.g., TankBind/P2Rank) and
experimentally curated catalytic residues (M-CSA), and assess robustness by perturbing the
predicted mask (drop/add a fraction of active residues) across multiple trials.

Outputs per-enzyme precision/recall/F1 and robustness statistics to a CSV.

Example usage:

python mcsa_sanitycheck.py \
  --dataset-path ../data/csv/train_dataset_hxkm_complex_unconditioned_bs.csv \
  --csa-path ../data/csa_subset.csv \
  --out-path ../data/mcsa_sanitycheck_and_robustness.csv \
  --mask-col conditioned_bs \
  --n-trials 30 --drop-frac 0.1 --add-frac 0.1
"""

import os
import ast
import argparse
import random
import numpy as np
import pandas as pd


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compare predicted binding-site masks to M-CSA catalytic residue positions and "
            "compute precision/recall/F1 + robustness via mask perturbation."
        )
    )

    p.add_argument("--dataset-path", type=str, required=True,
                   help="CSV with columns: uniprot_key, sequence, and a mask column (e.g. conditioned_bs).")
    p.add_argument("--csa-path", type=str, required=True,
                   help="CSV with columns: uniprot_key, catalytic_positions (1-based positions separated by ; or ,).")
    p.add_argument("--out-path", type=str, required=True,
                   help="Where to write the output CSV (per-enzyme metrics).")

    p.add_argument("--mask-col", type=str, default="conditioned_bs",
                   help="Column name in dataset CSV containing the binary mask (stored as a python-like list string).")

    p.add_argument("--n-trials", type=int, default=30,
                   help="Number of perturbation trials per enzyme.")
    p.add_argument("--drop-frac", type=float, default=0.10,
                   help="Fraction of predicted active residues to drop per trial.")
    p.add_argument("--add-frac", type=float, default=0.10,
                   help="Fraction of predicted active residues (original size) to add per trial.")

    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--subset-to-csa", action="store_true",
                   help="If set, restrict dataset rows to UniProt IDs present in the CSA file.")
    p.add_argument("--verbose", action="store_true",
                   help="If set, print per-enzyme debug info (can be very noisy).")

    return p


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_positions(s):
    # expects "45;87;201" (1-based)
    if pd.isna(s) or str(s).strip() == "":
        return []
    s = str(s).replace(",", ";")
    return [int(x) for x in s.split(";") if x.strip().isdigit()]


def metrics(pred_set, true_set):
    tp = len(pred_set & true_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(true_set) if true_set else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1, tp


def perturb_mask(active_idxs, L, drop_frac=0.1, add_frac=0.1, rng=None):
    rng = rng or np.random.default_rng(0)
    active = set(active_idxs)

    # drop
    k_drop = int(round(drop_frac * len(active)))
    if k_drop > 0 and len(active) > 0:
        drop = rng.choice(list(active), size=min(k_drop, len(active)), replace=False)
        for i in drop:
            active.discard(int(i))

    # add
    comp = [i for i in range(L) if i not in active]
    k_add = int(round(add_frac * len(active_idxs)))  # add relative to original active size
    if k_add > 0 and len(comp) > 0:
        add = rng.choice(comp, size=min(k_add, len(comp)), replace=False)
        for i in add:
            active.add(int(i))

    return active


def med_iqr(x):
    x = x.dropna()
    return float(np.median(x)), float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.dataset_path)
    csa = pd.read_csv(args.csa_path)

    if args.subset_to_csa:
        csa_unis = set(str(u) for u in csa["uniprot_key"].unique())
        df = df[df["uniprot_key"].astype(str).isin(csa_unis)].reset_index(drop=True)

    # map UniProt -> catalytic set (0-based)
    csa_map = {}
    for _, r in csa.iterrows():
        uni = str(r["uniprot_key"])
        pos1 = parse_positions(r["catalytic_positions"])
        csa_map[uni] = {p - 1 for p in pos1 if p > 0}

    rows = []

    for _, r in df.iterrows():
        uni = str(r["uniprot_key"])
        if uni not in csa_map:
            continue

        seq = str(r["sequence"])
        L = len(seq)
        true_set = {i for i in csa_map[uni] if 0 <= i < L}

        # parse mask column (binary list as string)
        try:
            mask = np.array(ast.literal_eval(str(r[args.mask_col])), dtype=int)
        except (ValueError, TypeError, KeyError):
            continue

        if len(mask) != L:
            continue

        pred_set = {int(i) for i in np.where(mask == 1)[0]}
        prec, rec, f1, tp = metrics(pred_set, true_set)

        if args.verbose:
            print(f"[{uni}] L={L} true={len(true_set)} pred={len(pred_set)} tp={tp} "
                  f"prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")

        # robustness: perturb mask and recompute overlap metrics
        precs, recs, f1s = [], [], []
        for _ in range(args.n_trials):
            pert = perturb_mask(list(pred_set), L, drop_frac=args.drop_frac, add_frac=args.add_frac, rng=rng)
            p2, r2, f2, _ = metrics(pert, true_set)
            precs.append(p2); recs.append(r2); f1s.append(f2)

        rows.append({
            "uniprot_key": uni,
            "L": L,
            "n_true": len(true_set),
            "n_pred": len(pred_set),
            "tp": tp,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "precision_pert_mean": float(np.mean(precs)),
            "precision_pert_std": float(np.std(precs)),
            "recall_pert_mean": float(np.mean(recs)),
            "recall_pert_std": float(np.std(recs)),
            "f1_pert_mean": float(np.mean(f1s)),
            "f1_pert_std": float(np.std(f1s)),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_path, index=False)

    # summary prints (kept concise)
    if len(out) == 0:
        print("No matching enzymes found or all rows were filtered out. Output CSV is empty.")
        print(f"Wrote: {args.out_path}")
        return

    df2 = out.drop_duplicates(subset=["uniprot_key"]).copy()
    df2 = df2[df2["n_true"] > 0]

    summary_cols = ["precision", "recall", "f1",
                    "precision_pert_mean", "recall_pert_mean", "f1_pert_mean",
                    "precision_pert_std", "recall_pert_std", "f1_pert_std"]

    print(f"N enzymes: {len(df2)}")
    print("n_true median (IQR):", med_iqr(df2["n_true"]))
    print("n_pred median (IQR):", med_iqr(df2["n_pred"]))

    print("\nKey overlap metrics (median [Q1,Q3])")
    for k in ["precision", "recall", "f1"]:
        m, q1, q3 = med_iqr(df2[k])
        print(f"{k}: {m:.3f} [{q1:.3f}, {q3:.3f}]")

    # deltas (robustness)
    df2["delta_f1"] = df2["f1_pert_mean"] - df2["f1"]
    df2["delta_recall"] = df2["recall_pert_mean"] - df2["recall"]
    df2["delta_precision"] = df2["precision_pert_mean"] - df2["precision"]

    print("\nRobustness deltas (median [Q1,Q3])")
    for k in ["delta_precision", "delta_recall", "delta_f1"]:
        m, q1, q3 = med_iqr(df2[k])
        print(f"{k}: {m:.3f} [{q1:.3f}, {q3:.3f}]")

    print(f"\nSaved: {args.out_path}")


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
