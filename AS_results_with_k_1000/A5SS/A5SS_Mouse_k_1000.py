import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

def main():
    folder = "../GCS_scores/Mouse_ss_score_k_1000"
    a5_df = pd.read_csv("../AS_data/mm10_A5.txt", sep="\t")
    a5_df["index"]    = -1
    a5_df["prob_5ss"] = -1.0
    a5_df["prob_3ss"] = -1.0

    sec5_re = re.compile(
        r"Sorted 5' Splice Sites \(High to Low Probability\):([\s\S]+?)Sorted 3' Splice Sites",
        re.MULTILINE
    )
    sec3_re = re.compile(
        r"Sorted 3' Splice Sites \(High to Low Probability\):([\s\S]+)",
        re.MULTILINE
    )
    pos_re = re.compile(r"Position\s*(\d+):\s*([0-9.eE+\-]+)")


    gene_to_index = {}
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        text = open(os.path.join(folder, fname), encoding="utf-8").read()
        mg = re.search(r"Gene\s*=\s*(\S+)", text)
        mi = re.search(r"_g_(\d+)\.txt", fname)
        if mg and mi:
            gene_to_index[mg.group(1)] = (int(mi.group(1)), text)

    
    for i, row in a5_df.iterrows():
        gene = row["gene"]
        ss5  = int(row["5ss_alternative"])
        ss3  = int(row["3ss_pairing"])
        if gene not in gene_to_index:
            continue

        idx, text = gene_to_index[gene]
        a5_df.at[i, "index"] = idx

        # 5'
        prob5 = -1.0
        m5 = sec5_re.search(text)
        if m5:
            for line in m5.group(1).splitlines():
                m = pos_re.match(line.strip())
                if m and int(m.group(1)) == ss5:
                    prob5 = float(m.group(2))
                    break

        # 3'
        prob3 = -1.0
        m3 = sec3_re.search(text)
        if m3:
            for line in m3.group(1).splitlines():
                m = pos_re.match(line.strip())
                if m and int(m.group(1)) == ss3:
                    prob3 = float(m.group(2))
                    break

        a5_df.at[i, "prob_5ss"] = prob5
        a5_df.at[i, "prob_3ss"] = prob3

    filtered = a5_df[(a5_df["prob_5ss"] > 0) & (a5_df["prob_3ss"] > 0)]
    indexes = sorted(filtered["index"].unique())

    records = []
    for idx in indexes:
        sub = filtered[filtered["index"] == idx]
        gene = sub["gene"].iloc[0]
        max_row = sub.loc[sub["PSI"].idxmax()]
        min_row = sub.loc[sub["PSI"].idxmin()]
        records.append({
            "index":    idx,
            "gene":     gene,
            "PSI":      max_row["PSI"],
            "prob_5ss": max_row["prob_5ss"],
            "type":     "major"
        })
        records.append({
            "index":    idx,
            "gene":     gene,
            "PSI":      min_row["PSI"],
            "prob_5ss": min_row["prob_5ss"],
            "type":     "minor"
        })
    df_extremes = pd.DataFrame(records)

    df_extremes = df_extremes[
        df_extremes.groupby("index")["prob_5ss"]
                   .transform(lambda x: x.nunique() > 1)
    ]

    stats_df = (
        df_extremes
        .groupby("type")["prob_5ss"]
        .agg(count="count", avg_prob="mean", std_prob="std")
        .reset_index()
    )

    r, p = pearsonr(df_extremes["PSI"], df_extremes["prob_5ss"])
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    sns.boxplot(
        x="type", y="prob_5ss", data=df_extremes,
        order=["major", "minor"],
        showfliers=False,
        boxprops=dict(facecolor="none", edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
        width=0.4,
        linewidth=5,
        ax=ax1
    )

    for idx, grp in df_extremes.groupby("index"):
        if grp["type"].nunique() == 2:
            y_major = grp.loc[grp["type"] == "minor", "prob_5ss"].values[0]
            y_minor = grp.loc[grp["type"] == "major", "prob_5ss"].values[0]
            ax1.plot([0, 1], [y_minor, y_major],
                     color="gray", linewidth=1.2, alpha=1, zorder=0)

    sns.swarmplot(
        x="type", y="prob_5ss",
        data=df_extremes,
        order=["major", "minor"],
        hue="type", palette={"minor":"lightgreen","major":"darkgreen"},
        size=2, ax=ax1, dodge=False
    )

    ax1.set_ylabel("5'SS Score", fontsize=40)
    ax1.set_xlabel("Mouse", fontsize=40, fontweight="bold")

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Major", "Minor"], fontsize=40)
    ax1.xaxis.set_tick_params(labelsize=40)
    ax1.yaxis.set_tick_params(labelsize=40)
    ax1.set_yticks([0, 1])


    jitter = 0.5
    x = df_extremes["PSI"] + np.random.normal(scale=jitter, size=len(df_extremes))
    y = df_extremes["prob_5ss"]
    colors = df_extremes["type"].map({"major":"darkgreen","minor":"lightgreen"})
    ax2.scatter(x, y, c=colors, s=50, alpha=1.0, edgecolors='none')
    ax2.set_xlabel("PSI", fontsize=40)
    ax2.set_ylabel("5'SS Score", fontsize=40)
    ax2.set_xticks([0, 100])
    ax2.set_yticks([0, 1])
    ax2.xaxis.set_tick_params(labelsize=40)
    ax2.yaxis.set_tick_params(labelsize=40)

    plt.tight_layout()
    plt.savefig("./A5SS_Mouse.png", dpi=300)
    print("###### Mouse (A5SS) ######")
    print(f"r = {r:.2f}\n(p = {p:.2g})")
    print("Figure saved as A5SS_Mouse.png")

if __name__ == "__main__":
    main()

