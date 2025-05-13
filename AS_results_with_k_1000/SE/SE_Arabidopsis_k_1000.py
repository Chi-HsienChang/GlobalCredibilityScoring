import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns



def main():
    folder = "../GCS_scores/Arabidopsis_exon_score_k_1000"
    AS_path = "../AS_data/tair10_SE.txt"
    se_df = pd.read_csv(AS_path, sep="\t")
    se_df["index"] = -1
    se_df["prob"] = -1.0

    gene_to_index = {}
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(folder, fname)
        content = open(path, encoding="utf-8").read()
        mg = re.search(r"Gene\s*=\s*(\S+)", content)
        mi = re.search(r"_g_(\d+)\.txt", fname)
        if mg and mi:
            gene_to_index[mg.group(1)] = (int(mi.group(1)), content)

    for i, row in se_df.iterrows():
        gene = row["gene"]
        ss_5, ss_3 = int(row["5ss"]), int(row["3ss"])
        if gene not in gene_to_index:
            continue
        idx, text = gene_to_index[gene]
        se_df.at[i, "index"] = idx

        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"^(\d+),\s*(\d+),\s*([0-9.eE+\-]+)$", line)
            if not m:
                continue
            three_val, five_val, p = int(m.group(1)), int(m.group(2)), float(m.group(3))
            if three_val == ss_3 and five_val == ss_5:
                se_df.at[i, "prob"] = p
                break

    filtered_df = se_df[se_df["prob"] != -1.0]
    available_indexes = set(filtered_df["index"].unique())

    se_info = {
        (r["gene"], r["5ss"], r["3ss"]): r["PSI"]
        for _, r in pd.read_csv(AS_path, sep="\t").iterrows()
    }

    records = []

    gene_re  = re.compile(r"Gene\s*=\s*(\S+)")
    idx_re   = re.compile(r"index\s*=\s*(\d+)")
    a5_re    = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    a3_re    = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
    line_re  = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.eE+\-]+)\s*$")

    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        content = open(os.path.join(folder, fname), encoding="utf-8").read()
        g_m = gene_re.search(content)
        i_m = idx_re.search(content)

        ann5 = [int(x) for x in re.findall(r"\d+", a5_re.search(content).group(1))] if a5_re.search(content) else []
        ann3 = [int(x) for x in re.findall(r"\d+", a3_re.search(content).group(1))] if a3_re.search(content) else []
        pair_count = min(len(ann3), len(ann5)) - 1

        for j in range(pair_count):
            three_ss = ann3[j]
            five_ss  = ann5[j+1]
            key = (g_m.group(1), five_ss, three_ss) if g_m else (None, None, None)
            classification = "SE" if key in se_info else "non-SE"
            psi_val = se_info.get(key)
            prob = None
            for line in content.splitlines():
                m = line_re.match(line.strip())
                if m and int(m.group(1)) == three_ss and int(m.group(2)) == five_ss:
                    prob = float(m.group(3))
                    break
            records.append({
                "gene": key[0],
                "5ss": five_ss,
                "3ss": three_ss,
                "PSI": psi_val,
                "index": int(i_m.group(1)) if i_m else None,
                "prob": prob,
                "classification": classification
            })

    df = pd.DataFrame(records).sort_values(by="index").reset_index(drop=True)
    df_filtered = df[df["index"].isin(available_indexes)]

    valid_df = df_filtered[
        (df_filtered["classification"] != "NA") &
        (df_filtered["prob"].notnull())
    ]
    summary = valid_df.groupby("classification").prob.agg(
        count="count", avg_prob="mean", std_prob="std"
    ).reset_index()


    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    sns.swarmplot(
        x="classification", y="prob", data=valid_df,
        hue="classification", legend=False, 
        ax=ax1, order=["non-SE", "SE"],
        palette={"SE": "#2E8B57", "non-SE": "blue"},
        dodge=False, size=5, linewidth=0
    )

    sns.boxplot(
        x="classification", y="prob", data=valid_df,
        hue="classification", legend=False, 
        ax=ax1, order=["non-SE", "SE"], showfliers=False,
        palette={"SE": "red", "non-SE": "black"},
        boxprops={"facecolor": "white", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black"},
        width=0.4,
        linewidth=5
    )

    ax1.set_xlabel("Arabidopsis", fontsize=40, fontweight="bold")
    ax1.set_ylabel("Exon Score", fontsize=40)
    ax1.tick_params(axis='x', labelsize=40) 
    ax1.tick_params(axis='y', labelsize=40) 
    ax1.set_yticks([0.0, 1.0])


    def kde_cdf_smooth(vals, grid_size=300):
        arr = np.array(vals)
        if arr.size == 0:
            return np.array([]), np.array([])
        x = np.linspace(arr.min(), arr.max(), grid_size)
        kde = gaussian_kde(arr)
        pdf = kde(x)
        cdf = np.cumsum(pdf) / np.sum(pdf)
        return x, cdf

    x_se,  cdf_se  = kde_cdf_smooth(valid_df[valid_df["classification"]=="SE"]["prob"].values)
    x_non, cdf_non = kde_cdf_smooth(valid_df[valid_df["classification"]=="non-SE"]["prob"].values)
    ax2.plot(x_se,  cdf_se,  label="SE",     color="#2E8B57",   linewidth=7)
    ax2.plot(x_non, cdf_non, label="non-SE", color="blue", linewidth=7)
    ax2.set_xlabel("Exon Score", fontsize=40)
    ax2.set_ylabel("eCDF", fontsize=40)
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0.0, 1.0])
    ax2.set_yticks([0.0, 1.0])
    ax2.tick_params(axis='x', labelsize=40) 
    ax2.tick_params(axis='y', labelsize=40)  

    ax2.legend(
        fontsize=38,
        frameon=False,
        loc="lower right",
        handlelength=0.2,      
        scatterpoints=1,      
        bbox_to_anchor=(1.07, -0.13)
    )

    plt.tight_layout()
    plt.savefig("./SE_Arabidopsis.png", dpi=300)
    print("###### Arabidopsis (SE) ######")
    print("Figure saved as SE_Arabidopsis.png")
    

if __name__ == "__main__":
    main()

