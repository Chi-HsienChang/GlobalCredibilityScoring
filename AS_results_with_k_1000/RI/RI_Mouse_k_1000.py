import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns



def main():
    folder = "../GCS_scores/Mouse_intron_score_k_1000"
    AS_path = "../AS_data/mm10_RI.txt"
    ri_df = pd.read_csv(AS_path, sep="\t")
    ri_df["index"] = -1
    ri_df["prob"] = -1.0


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

    line_re = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.eE+\-]+)\s*$")
    for i, row in ri_df.iterrows():
        gene = row["gene"]
        s5, s3 = int(row["5ss"]), int(row["3ss"])
        if gene not in gene_to_index:
            continue
        
        idx, text = gene_to_index[gene]
        ri_df.at[i, "index"] = idx

        for line in text.splitlines():
            m = line_re.match(line)
            if not m:
                continue
            five_val, three_val, p = int(m.group(1)), int(m.group(2)), float(m.group(3))
            if five_val == s5 and three_val == s3:
                ri_df.at[i, "prob"] = p
                break

    filtered_df = ri_df[ri_df["prob"] != -1.0]
    available_indexes = set(filtered_df["index"].unique())

    ri_info = {
        (r["gene"], r["5ss"], r["3ss"]): r["PSI"]
        for _, r in pd.read_csv(AS_path, sep="\t").iterrows()
    }

    records = []
    gene_re    = re.compile(r"Gene\s*=\s*(\S+)")
    idx_re     = re.compile(r"index\s*=\s*(\d+)")
    ann5_re    = re.compile(r"Annotated\s+5SS:\s*\[([^\]]*)\]")
    ann3_re    = re.compile(r"Annotated\s+3SS:\s*\[([^\]]*)\]")

    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        content = open(os.path.join(folder, fname), encoding="utf-8").read()
        g_m = gene_re.search(content)
        i_m = idx_re.search(content)
        ann5 = [int(x) for x in re.findall(r"\d+", ann5_re.search(content).group(1))] if ann5_re.search(content) else []
        ann3 = [int(x) for x in re.findall(r"\d+", ann3_re.search(content).group(1))] if ann3_re.search(content) else []
        pair_count = min(len(ann5), len(ann3))

        for j in range(pair_count):
            five_ss, three_ss = ann5[j], ann3[j]
            key = (g_m.group(1), five_ss, three_ss) if g_m else (None, None, None)
            classification = "RI" if key in ri_info else "non-RI"
            psi_val = ri_info.get(key)
            prob = None
            for line in content.splitlines():
                m = line_re.match(line.strip())
                if m and int(m.group(1)) == five_ss and int(m.group(2)) == three_ss:
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

    df = pd.DataFrame(records).sort_values("index").reset_index(drop=True)


    df_filtered = df[df["index"].isin(available_indexes)]

    valid_df = df_filtered[
        (df_filtered["classification"] != "NA") &
        (df_filtered["prob"].notnull())
    ]
    summary = valid_df.groupby("classification")["prob"].agg(
        count="count", avg_prob="mean", std_prob="std"
    ).reset_index()

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    sns.swarmplot(
        x="classification", y="prob", data=valid_df,
        hue="classification", legend=False,  
        ax=ax1, order=["non-RI", "RI"],
        palette={"RI": "#2E8B57", "non-RI": "blue"},
        dodge=False, size=1.5, linewidth=0
    )

    sns.boxplot(
        x="classification", y="prob", data=valid_df,
        hue="classification", legend=False, 
        ax=ax1, order=["non-RI", "RI"], showfliers=False,
        palette={"RI": "red", "non-RI": "black"},
        boxprops={"facecolor": "white", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black"},
        width=0.4,
        linewidth=5
    )

    ax1.set_xlabel("Mouse", fontsize=40, fontweight="bold")

    ax1.set_ylabel("Intron Score", fontsize=40)
    ax1.set_yticks([0.0, 1.0])
    ax1.tick_params(axis='x', labelsize=40)  
    ax1.tick_params(axis='y', labelsize=40)  


    def kde_cdf_smooth(vals, grid_size=300):
        arr = np.array(vals)
        if arr.size == 0:
            return np.array([]), np.array([])
        x = np.linspace(arr.min(), arr.max(), grid_size)
        kde = gaussian_kde(arr)
        pdf = kde(x)
        cdf = np.cumsum(pdf) / np.sum(pdf)
        return x, cdf

    x_ri, cdf_ri     = kde_cdf_smooth(valid_df[valid_df["classification"]=="RI"]["prob"])
    x_nonri, cdf_nonri = kde_cdf_smooth(valid_df[valid_df["classification"]=="non-RI"]["prob"])
    ax2.plot(x_ri, cdf_ri, label="RI", color="#2E8B57", linewidth=12)
    ax2.plot(x_nonri, cdf_nonri, label="non-RI", color="blue", linewidth=12)
    ax2.set_xlabel("Intron Score", fontsize=40)
    ax2.set_ylabel("eCDF", fontsize=40)
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0.0, 1.0])
    ax2.set_yticks([0.0, 1.0])
    ax2.tick_params(axis='x', labelsize=40)  
    ax2.tick_params(axis='y', labelsize=40)  

    ax2.legend(
        fontsize=40,
        frameon=False,
        loc="lower right",
        handlelength=0.2,       
        scatterpoints=1,     
        bbox_to_anchor=(1.07, -0.13)
    )

    plt.tight_layout()
    plt.savefig("./RI_Mouse.png", dpi=300)
    print("###### Mouse (RI) ######")
    print("Figure saved as RI_Mouse.png")

if __name__ == "__main__":
    main()

