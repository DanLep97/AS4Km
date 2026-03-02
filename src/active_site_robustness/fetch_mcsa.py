"""
Fetch M-CSA catalytic residue positions for UniProt IDs present in a dataset CSV.

The script:
1) reads a dataset CSV containing a `uniprot_key` column,
2) queries the M-CSA residues API for each UniProt (canonicalized to remove isoform suffix),
3) extracts reference UniProt residue positions (1-based),
4) writes a subset CSV with columns: uniprot_key, catalytic_positions, n_catalytic_residues.

Example usage
-------------
python fetch_mcsa_subset.py \
    --dataset-path data/csv/train_dataset_hxkm_complex_conditioned_bs.csv \
    --out-path data/csv/csa_subset.csv \
    --n-target 100 \
    --sleep-seconds 0.25 \
    --seed 42
"""

import argparse
import random
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Any, Optional

import requests
import pandas as pd


MCSA_RESIDUES_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Query the M-CSA residues API for UniProt IDs from a dataset and write a CSV "
            "containing catalytic residue positions (1-based)."
        )
    )
    p.add_argument("--dataset-path", type=str, required=True,
                   help="Input CSV containing a `uniprot_key` column.")
    p.add_argument("--out-path", type=str, required=True,
                   help="Output CSV path for the M-CSA subset.")
    p.add_argument("--uniprot-col", type=str, default="uniprot_key",
                   help="Column name in dataset CSV containing UniProt IDs.")
    p.add_argument("--n-target", type=int, default=100,
                   help="Maximum number of unique UniProt entries to write.")
    p.add_argument("--sleep-seconds", type=float, default=0.25,
                   help="Sleep between API requests to avoid rate-limiting.")
    p.add_argument("--timeout", type=int, default=30,
                   help="Requests timeout in seconds.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for shuffling UniProts.")
    p.add_argument("--resume", action="store_true",
                   help="If out-path exists, load it and only query UniProts not already present.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-UniProt progress messages.")
    return p


def canonical_uniprot(u: str) -> str:
    """Strip whitespace and isoform suffix (P12345-2 -> P12345)."""
    u = str(u).strip()
    return u.split("-")[0]


def fetch_mcsa_residues_for_uniprot(
    uniprot_id: str,
    session: requests.Session,
    timeout: int,
) -> List[Any]:
    """
    Returns a list of residue JSON objects for this UniProt.
    Empty list means: no curated M-CSA residues found (or no match).
    """
    params = {
        "format": "json",
        "entries.proteins.sequences.uniprot_ids": uniprot_id,
    }
    r = session.get(MCSA_RESIDUES_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    # API usually returns a list. Handle common wrappers defensively.
    if isinstance(data, dict):
        for k in ("results", "data", "items"):
            if k in data and isinstance(data[k], list):
                return data[k]
        return []
    if isinstance(data, list):
        return data
    return []


def extract_positions_from_payload(payload: List[Any]) -> Dict[str, Set[int]]:
    """
    Extract UniProt positions from residue_sequences[*].resid where is_reference==True.
    Returns: {uniprot_id: set(positions)} with 1-based residue positions.
    """
    out: Dict[str, Set[int]] = defaultdict(set)

    for obj in payload:
        for rs in obj.get("residue_sequences", []) or []:
            if not rs.get("is_reference", False):
                continue
            uni = rs.get("uniprot_id", None)
            pos = rs.get("resid", None)
            if uni is None or pos is None:
                continue
            try:
                out[str(uni)].add(int(pos))
            except Exception:
                pass

    return out


def load_existing_out(out_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(out_path)
    except Exception:
        return pd.DataFrame(columns=["uniprot_key", "catalytic_positions", "n_catalytic_residues"])


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    df = pd.read_csv(args.dataset_path)
    if args.uniprot_col not in df.columns:
        raise ValueError(f"Column '{args.uniprot_col}' not found in dataset CSV: {args.dataset_path}")

    uniprots = [canonical_uniprot(u) for u in df[args.uniprot_col].dropna().unique().tolist()]
    random.shuffle(uniprots)

    existing = pd.DataFrame()
    already_have: Set[str] = set()
    hits: List[dict] = []

    if args.resume:
        existing = load_existing_out(args.out_path)
        if "uniprot_key" in existing.columns:
            already_have = set(existing["uniprot_key"].astype(str).tolist())
        if len(existing) > 0 and args.verbose:
            print(f"Resume enabled: loaded {len(existing)} existing rows from {args.out_path}")

    # Use a session for connection pooling + add a polite User-Agent
    session = requests.Session()
    session.headers.update({
        "User-Agent": "mcsa-fetch-script/1.0 (contact: your-email-or-lab; purpose: research)"
    })

    seen: Set[str] = set()
    total_queried = 0

    for i, uni in enumerate(uniprots):
        if len(already_have) + len(hits) >= args.n_target:
            break
        if uni in seen or uni in already_have:
            continue
        seen.add(uni)

        total_queried += 1
        try:
            payload = fetch_mcsa_residues_for_uniprot(uni, session=session, timeout=args.timeout)
        except Exception as e:
            if args.verbose:
                print(f"[{i}] {uni} -> ERROR {e}")
            time.sleep(args.sleep_seconds)
            continue

        if not payload:
            if args.verbose:
                print(f"[{i}] {uni} -> no M-CSA")
            time.sleep(args.sleep_seconds)
            continue

        pos_map = extract_positions_from_payload(payload)

        found_any = False
        for uni_returned, positions in pos_map.items():
            if not positions:
                continue
            found_any = True
            hits.append({
                "uniprot_key": str(uni_returned),
                "catalytic_positions": ";".join(map(str, sorted(positions))),  # 1-based
                "n_catalytic_residues": int(len(positions)),
            })
            if args.verbose:
                print(f"[{i}] query={uni} -> HIT uni={uni_returned} (n={len(positions)}) "
                      f"| total={(len(already_have) + len(hits))}/{args.n_target}")

            if len(already_have) + len(hits) >= args.n_target:
                break

        if args.verbose and not found_any:
            print(f"[{i}] {uni} -> payload returned but no reference residue positions parsed")

        time.sleep(args.sleep_seconds)

    out_new = pd.DataFrame(hits).drop_duplicates(subset=["uniprot_key"])
    if args.resume and len(existing) > 0:
        out = pd.concat([existing, out_new], ignore_index=True)
        out = out.drop_duplicates(subset=["uniprot_key"])
    else:
        out = out_new

    out = out.head(args.n_target)
    out.to_csv(args.out_path, index=False)

    print(f"Saved {len(out)} entries to {args.out_path}")
    print(f"Queried {total_queried} UniProt IDs from dataset (resume={args.resume}).")
    print(out.head())


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
