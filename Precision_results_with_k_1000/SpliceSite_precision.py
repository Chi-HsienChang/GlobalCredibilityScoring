import os
import re
import glob
import pandas as pd
import numpy as np
from typing import Tuple, Set, List

# ----------------------------------------------------------------------------
# Configurable parameters
# ----------------------------------------------------------------------------
DATASET_NAME = "Arabidopsis"
TOP_K = 1000
BASE_DIR = f"./{DATASET_NAME}_SpliceSiteScore_k_{TOP_K}"
FILE_PATTERN = os.path.join(BASE_DIR, "arabidopsis_g_*.txt")

SHOW_BINS: List[str] = [
    "[0.9, 1.0]",
    "[0.8,0.9)",
    "[0.7,0.8)",
    "[0.6,0.7)",
    "[0.5,0.6)",
    "[0.4,0.5)",
]
# ----------------------------------------------------------------------------

# Regex helpers (compiled once)
ANN5_RE = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
ANN3_RE = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
VIT5_RE = re.compile(r"SMsplice 5SS:\s*\[([^\]]*)\]")
VIT3_RE = re.compile(r"SMsplice 3SS:\s*\[([^\]]*)\]")
LINE_RE = re.compile(r"Position\s+(\d+)\s*:\s*([\d.eE+-]+)")
BLOCK5_RE = re.compile(
    r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)", re.S
)
BLOCK3_RE = re.compile(r"Sorted 3['′] Splice Sites .*?\n(.*)", re.S)

# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------

def _as_int_set(match: re.Match) -> Set[int]:
    """Convert a regex match with comma/space separated numbers into a set."""
    if not match or not match.group(1).strip():
        return set()
    return set(map(int, re.split(r"[\s,]+", match.group(1).strip())))


def _preds(block: str) -> List[Tuple[int, float]]:
    return [(int(p), float(pb)) for p, pb in LINE_RE.findall(block)]


def parse_splice_file(path: str) -> Tuple[pd.DataFrame, Set[int], Set[int], Set[int]]:
    """Return (df, ground_truth_positions, vit5_set, vit3_set)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    ann5 = _as_int_set(ANN5_RE.search(text))
    ann3 = _as_int_set(ANN3_RE.search(text))
    vit5 = _as_int_set(VIT5_RE.search(text))
    vit3 = _as_int_set(VIT3_RE.search(text))

    preds5 = _preds(BLOCK5_RE.search(text).group(1)) if BLOCK5_RE.search(text) else []
    preds3 = _preds(BLOCK3_RE.search(text).group(1)) if BLOCK3_RE.search(text) else []

    rows = [(pos, prob, "5prime", pos in ann5, pos in vit5) for pos, prob in preds5]
    rows += [(pos, prob, "3prime", pos in ann3, pos in vit3) for pos, prob in preds3]

    df = pd.DataFrame(
        rows,
        columns=["position", "prob", "type", "is_correct", "in_viterbi"],
    )
    return df, ann5.union(ann3), vit5, vit3

# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def collect_predictions(files: List[str]) -> pd.DataFrame:
    """Return DataFrame with columns prob and is_correct for *all* Viterbi sites."""
    recs: List[Tuple[float, bool]] = []
    print(f"Processing Arabidopsis with top_k = 1000 for splice site scores...")
    for file_path in files:

        print(f"Processing {os.path.basename(file_path)}...")
        df, ground_truth, vit5, vit3 = parse_splice_file(file_path)
        have_prob_positions = set(df["position"].tolist())

        # Existing probability rows (already filtered to Viterbi sites)
        for _, row in df[df["in_viterbi"]].iterrows():
            prob = row["prob"]
            recs.append((prob, row["position"] in ground_truth))

        # Add missing Viterbi positions with dummy scores
        missing5 = vit5 - have_prob_positions
        missing3 = vit3 - have_prob_positions
        recs.extend([(-1.0, p in ground_truth) for p in missing5])
        recs.extend([(-0.5, p in ground_truth) for p in missing3])

    res = pd.DataFrame(recs, columns=["prob", "is_correct"])
    res["prob"] = res["prob"].clip(upper=0.99)

    print(f"Total number of txt files: {len(files)}")
    return res

# ----------------------------------------------------------------------------
# Bin statistics
# ----------------------------------------------------------------------------

def get_bin_info(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compute N_in_bin, Precision (%), and percentage (% of total) for bins."""
    edges = np.array([-1.0, 0.0] + list(np.arange(0.1, 0.9, 0.1)) + [0.9, 1.0])
    labels = [
        "0",
        "(0.0,0.1)",
        "[0.1,0.2)",
        "[0.2,0.3)",
        "[0.3,0.4)",
        "[0.4,0.5)",
        "[0.5,0.6)",
        "[0.6,0.7)",
        "[0.7,0.8)",
        "[0.8,0.9)",
        "[0.9, 1.0]",
    ]

    tmp = pred_df.copy()
    tmp["bin"] = pd.cut(tmp["prob"], bins=edges, labels=labels, right=False)
    info = (
        tmp.groupby("bin", observed=False)  # explicit observed=False avoids FutureWarning
        .agg(N_in_bin=("prob", "count"), N_correct=("is_correct", "sum"))
        .reindex(labels, fill_value=0)
    )

    info["Precision"] = (info["N_correct"] / info["N_in_bin"]).fillna(0) * 100
    total = info["N_in_bin"].sum()
    info["percentage"] = info["N_in_bin"] / total * 100
    return info

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    files = sorted(glob.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files match {FILE_PATTERN}")

    bin_info = get_bin_info(collect_predictions(files))

    print("====== Splice Site Score Result ======")
    print(f"species = {DATASET_NAME}")
    print(f"Top_k   = {TOP_K}")
    print("precision:")

    for rng in SHOW_BINS:
        prec = bin_info.loc[rng, "Precision"] / 100.0  
        pct = bin_info.loc[rng, "percentage"]          
        print(f"{rng}  (Subset%)")
        print(f"{prec:.6f}  {pct:.6f}")


if __name__ == "__main__":
    main()
