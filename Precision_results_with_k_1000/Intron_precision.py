import re
import glob
import pandas as pd
import os 


# ====== Helper functions ======
def parse_splice_file(filename):
    """Parse splice site info from a single text file."""
    with open(filename, "r") as f:
        text = f.read()

    pattern_5ss = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    pattern_3ss = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
    pattern_smsplice_5ss = re.compile(r"SMsplice 5SS:\s*\[([^\]]*)\]")
    pattern_smsplice_3ss = re.compile(r"SMsplice 3SS:\s*\[([^\]]*)\]")

    def parse_list(regex):
        match = regex.search(text)
        if not match:
            return set()
        inside = match.group(1).strip()
        if not inside:
            return set()
        items = re.split(r"[\s,]+", inside.strip())
        return set(map(int, items))

    annotated_5prime = parse_list(pattern_5ss)
    annotated_3prime = parse_list(pattern_3ss)
    viterbi_5prime = parse_list(pattern_smsplice_5ss)
    viterbi_3prime = parse_list(pattern_smsplice_3ss)

    # Optionally parse the "Sorted 5' / 3' Splice Sites" blocks if needed
    pattern_5prime_block = re.compile(
        r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)",
        re.DOTALL
    )
    pattern_3prime_block = re.compile(
        r"Sorted 3['′] Splice Sites .*?\n(.*)",
        re.DOTALL
    )
    pattern_line = re.compile(r"Position\s+(\d+)\s*:\s*([\d.eE+-]+)")

    def parse_predictions(block_regex):
        """Parse lines `Position <pos>: <prob>` if found."""
        match_block = block_regex.search(text)
        if not match_block:
            return []
        block = match_block.group(1)
        return [
            (int(m.group(1)), float(m.group(2)))
            for m in pattern_line.finditer(block)
        ]

    fiveprime_preds = parse_predictions(pattern_5prime_block)
    threeprime_preds = parse_predictions(pattern_3prime_block)

    # Convert predictions into a DataFrame
    rows = []
    for (pos, prob) in fiveprime_preds:
        rows.append((pos, prob, "5prime", pos in annotated_5prime, pos in viterbi_5prime))
    for (pos, prob) in threeprime_preds:
        rows.append((pos, prob, "3prime", pos in annotated_3prime, pos in viterbi_3prime))

    # Add missing predictions with prob=0 if they appear in annotated or viterbi sets
    existing_5ss = {r[0] for r in rows if r[2] == "5prime"}
    existing_3ss = {r[0] for r in rows if r[2] == "3prime"}

    for pos in annotated_5prime - existing_5ss:
        rows.append((pos, 0.0, "5prime", True,  pos in viterbi_5prime))
    for pos in annotated_3prime - existing_3ss:
        rows.append((pos, 0.0, "3prime", True, pos in viterbi_3prime))
    for pos in viterbi_5prime - existing_5ss:
        rows.append((pos, 0.0, "5prime", False, True))
    for pos in viterbi_3prime - existing_3ss:
        rows.append((pos, 0.0, "3prime", False, True))

    return pd.DataFrame(
        rows,
        columns=["position", "prob", "type", "is_correct", "is_viterbi"]
    )

def method_column(method):
    """Return the correct DataFrame column name for the given method."""
    return "is_correct" if method == "annotated" else "is_viterbi"

def prob5SS(pos, df, method="annotated"):
    """Return the 5' site probability at position `pos` for the chosen method."""
    col = method_column(method)
    row = df[
        (df["type"] == "5prime") & (df[col] == True) & (df["position"] == pos)
    ]
    return row.iloc[0]["prob"] if not row.empty else 0.0

def prob3SS(pos, df, method="annotated"):
    """Return the 3' site probability at position `pos` for the chosen method."""
    col = method_column(method)
    row = df[
        (df["type"] == "3prime") & (df[col] == True) & (df["position"] == pos)
    ]
    return row.iloc[0]["prob"] if not row.empty else 0.0

# ====== Main Batch Processing ======
species_map = {
    'Arabidopsis': 'arabidopsis',
}

   
top_ks = [1000]  

results = []
all_intron_scores = []   # 用來記錄原始 intron 預測分數


for top_k in top_ks:
    for code, name in species_map.items():
        print(f"Processing {code} with top_k = {top_k} for intron scores...")
        pattern = (
            f"Arabidopsis_IntronScore_k_1000/{name}_g_*.txt"
        )
        file_list = sorted(glob.glob(pattern))
        if not file_list:
            continue

        correct_intron_scores = []
        incorrect_intron_scores = []

        for txt_file in file_list:
            print(f"Processing {os.path.basename(txt_file)}...")
            with open(txt_file) as f:
                content = f.read()

            match = re.search(r"Annotated 5SS:\s*(\[[^\]]*\])", content)
            if not match:
                continue
            ann5ss = list(map(int, re.findall(r'\d+', match.group(1))))

            match = re.search(r"Annotated 3SS:\s*(\[[^\]]*\])", content)
            if not match:
                continue
            ann3ss = list(map(int, re.findall(r'\d+', match.group(1))))

            match = re.search(r"SMsplice 5SS:\s*(\[[^\]]*\])", content)
            if not match:
                continue
            sm5ss = list(map(int, re.findall(r'\d+', match.group(1))))

            match = re.search(r"SMsplice 3SS:\s*(\[[^\]]*\])", content)
            if not match:
                continue
            sm3ss = list(map(int, re.findall(r'\d+', match.group(1))))

            ann_introns = set(
                (ann5ss[i], ann3ss[i])
                for i in range(min(len(ann5ss), len(ann3ss)))
            )
            sm_introns = set(
                (sm5ss[i], sm3ss[i])
                for i in range(min(len(sm5ss), len(sm3ss)))
            )

            df_splice = parse_splice_file(txt_file)

            intron_table = {}
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if re.search(r"5SS,\s*3SS,\s*prob", line, re.IGNORECASE):
                    for subsequent_line in lines[i+1:]:
                        subsequent_line = subsequent_line.strip()
                        if not subsequent_line:
                            break
                        parts = subsequent_line.split(',')
                        if len(parts) >= 3:
                            try:
                                intron_5 = int(parts[0].strip())
                                intron_3 = int(parts[1].strip())
                                intron_prob = float(parts[2].strip())
                                intron_table[(intron_5, intron_3)] = intron_prob
                            except Exception:
                                continue
                    break

            for (five_site, three_site) in sm_introns:
                p5 = prob5SS(five_site, df_splice, method="smsplice")
                p3 = prob3SS(three_site, df_splice, method="smsplice")

                if (five_site, three_site) in intron_table:
                    score = intron_table[(five_site, three_site)]
                else:
                    score = p5 * p3

                if score == 0.0:
                    continue

                is_correct = ((five_site, three_site) in ann_introns)
                if is_correct:
                    correct_intron_scores.append(score)
                    all_intron_scores.append({
                        'top_k': top_k,
                        'species': code,
                        'label': 'correct',
                        'score': score
                    })
                else:
                    incorrect_intron_scores.append(score)
                    all_intron_scores.append({
                        'top_k': top_k,
                        'species': code,
                        'label': 'incorrect',
                        'score': score
                    })

        total_preds = len(correct_intron_scores) + len(incorrect_intron_scores)
        if total_preds == 0:
            precision_09 = None
            subset_09 = None
        else:
            all_scores = correct_intron_scores + incorrect_intron_scores
            scores_above_09 = [s for s in all_scores if s >= 0.9]
            correct_above_09 = [s for s in correct_intron_scores if s >= 0.9]

            if len(scores_above_09) > 0:
                precision_09 = len(correct_above_09) / len(scores_above_09)
                subset_09 = len(scores_above_09) / total_preds
            else:
                precision_09 = None
                subset_09 = 0.0

        results.append({
            "top_k": top_k,
            "species": code,
            "precision_09": precision_09,
            "subset_09": subset_09*100
        })

        print(f"Total number of txt files: {len(file_list)}")


df_result = pd.DataFrame(results)
df_result = df_result.rename(columns={
    "precision_09": "precision with >=0.9",
    "subset_09": "(Subset%)"
})
print("====== Intron Score Result ======")
print(df_result.to_string(index=False, float_format="{:.6f}".format))



