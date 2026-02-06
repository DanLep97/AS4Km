import os
import numpy as np
import pandas as pd
import random
import ast

# set random seed for reproducibility
np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

def parse_positions(s):
    # expects "45;87;201" (1-based)
    if pd.isna(s) or str(s).strip() == "":
        return []
    s = str(s).replace(",", ";")
    return [int(x) for x in s.split(";") if x.strip().isdigit()]

def metrics(pred_set, true_set):
    tp = len(pred_set & true_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec  = tp / len(true_set) if true_set else 0.0
    f1 = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
    print("TP:", tp, "TP + FP:", len(pred_set), "TP + FN:", len(true_set))
    return prec, rec, f1

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

# ---- load inputs ----
dataset_path = "path/to/dataset.csv"         # has uniprot_key, sequence, unconditioned_bs
csa_path = "path/to/csa_subset.csv"               # has uniprot_key, catalytic_positions

df = pd.read_csv(dataset_path)
csa = pd.read_csv(csa_path)

# # subset df to uniprots in csa
# csa_unis = set(str(u) for u in csa["uniprot_key"].unique())
# df = df[df["uniprot_key"].apply(lambda u: str(u) in csa_unis)].reset_index(drop=True)

# map UniProt -> catalytic set (0-based)
csa_map = {}
for _, r in csa.iterrows():
    uni = str(r["uniprot_key"])
    pos1 = parse_positions(r["catalytic_positions"])
    csa_map[uni] = set([p-1 for p in pos1 if p > 0])

rng = np.random.default_rng(42)
rows = []

N_TRIALS = 30
DROP = 0.10
ADD = 0.10

for _, r in df.iterrows():
    uni = str(r["uniprot_key"])
    if uni not in csa_map:
        continue

    seq = str(r["sequence"])
    L = len(seq)
    true_set = {i for i in csa_map[uni] if 0 <= i < L}
    print("True set:", true_set)

    # TankBind mask (binary list as string)
    try:
        mask = np.array(ast.literal_eval(str(r["conditioned_bs"])), dtype=int)
    except (ValueError, TypeError):
        # Skip rows with invalid or missing conditioned_bs values
        continue
    if len(mask) != L:
        continue

    pred_set = {int(i) for i in np.where(mask == 1)[0]}
    print("Pred set:", pred_set)
    prec, rec, f1 = metrics(pred_set, true_set)

    # robustness: perturb mask and recompute overlap metrics
    precs, recs, f1s = [], [], []
    for _ in range(N_TRIALS):
        pert = perturb_mask(list(pred_set), L, drop_frac=DROP, add_frac=ADD, rng=rng)
        p2, r2, f2 = metrics(pert, true_set)
        precs.append(p2); recs.append(r2); f1s.append(f2)

    rows.append({
        "uniprot_key": uni,
        "L": L,
        "n_true": len(true_set),
        "n_pred": len(pred_set),
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
out.to_csv("path/to/mcsa_sanitycheck_and_robustness_p2rank_tb.csv", index=False)

print(out.describe(include="all"))
print("Saved: path/to/mcsa_sanitycheck_and_robustness_p2rank_tb.csv")

# path = "path/to/mcsa_sanitycheck_and_robustness_p2rank_tb.csv"
# df = pd.read_csv(path)

df = out

# drop duplicates
df = df.drop_duplicates(subset=["uniprot_key"]).copy()

# sanity filters
df = df[df["n_true"] > 0]

def med_iqr(x):
    x = x.dropna()
    return float(np.median(x)), float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))

summary = {}
for col in ["precision", "recall", "f1",
            "precision_pert_mean", "recall_pert_mean", "f1_pert_mean",
            "precision_pert_std", "recall_pert_std", "f1_pert_std"]:
    m, q1, q3 = med_iqr(df[col])
    summary[col] = {"median": m, "q1": q1, "q3": q3}

# deltas (robustness)
df["delta_f1"] = df["f1_pert_mean"] - df["f1"]
df["delta_recall"] = df["recall_pert_mean"] - df["recall"]
df["delta_precision"] = df["precision_pert_mean"] - df["precision"]

for col in ["delta_precision", "delta_recall", "delta_f1"]:
    m, q1, q3 = med_iqr(df[col])
    summary[col] = {"median": m, "q1": q1, "q3": q3}

print("N enzymes:", len(df))
print("n_true median (IQR):", med_iqr(df["n_true"]))
print("n_pred median (IQR):", med_iqr(df["n_pred"]))

print("\nKey overlap metrics (median [Q1,Q3])")
for k in ["precision", "recall", "f1"]:
    s = summary[k]
    print(f"{k}: {s['median']:.3f} [{s['q1']:.3f}, {s['q3']:.3f}]")

print("\nRobustness deltas (median [Q1,Q3])")
for k in ["delta_precision", "delta_recall", "delta_f1"]:
    s = summary[k]
    print(f"{k}: {s['median']:.3f} [{s['q1']:.3f}, {s['q3']:.3f}]")
