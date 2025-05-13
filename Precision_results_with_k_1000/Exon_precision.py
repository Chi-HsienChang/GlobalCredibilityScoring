import re
import glob
import pandas as pd
import os 

# ====== Helper functions ======
def parse_splice_file(filename):
    with open(filename, "r") as f:
        text = f.read()

    # Regex patterns for 3SS / 5SS (annotated and SMsplice)
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

    pattern_5prime_block = re.compile(
        r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)", 
        re.DOTALL
    )
    pattern_3prime_block = re.compile(
        r"Sorted 3['′] Splice Sites .*?\n(.*)", 
        re.DOTALL
    )
    pattern_line = re.compile(r"Position\s+(\d+)\s*:\s*([\d.eE+-]+)")

    def parse_predictions(pattern):
        match_block = pattern.search(text)
        if not match_block:
            return []
        block = match_block.group(1)
        return [(int(m.group(1)), float(m.group(2))) for m in pattern_line.finditer(block)]

    fiveprime_preds = parse_predictions(pattern_5prime_block)
    threeprime_preds = parse_predictions(pattern_3prime_block)

    rows = []
    for (pos, prob) in fiveprime_preds:
        rows.append((pos, prob, "5prime", pos in annotated_5prime, pos in viterbi_5prime))
    for (pos, prob) in threeprime_preds:
        rows.append((pos, prob, "3prime", pos in annotated_3prime, pos in viterbi_3prime))

    existing_5ss = {r[0] for r in rows if r[2] == "5prime"}
    existing_3ss = {r[0] for r in rows if r[2] == "3prime"}

    for pos in annotated_5prime - existing_5ss:
        rows.append((pos, 0.0, "5prime", True, pos in viterbi_5prime))
    for pos in annotated_3prime - existing_3ss:
        rows.append((pos, 0.0, "3prime", True, pos in viterbi_3prime))
    for pos in viterbi_5prime - existing_5ss:
        rows.append((pos, 0.0, "5prime", False, True))
    for pos in viterbi_3prime - existing_3ss:
        rows.append((pos, 0.0, "3prime", False, True))

    return pd.DataFrame(rows, columns=["position", "prob", "type", "is_correct", "is_viterbi"])

def method_column(method):
    return "is_correct" if method == "annotated" else "is_viterbi"

def prob3SS(pos, df, method="annotated"):
    row = df[
        (df["type"]=="3prime") & 
        (df[method_column(method)]==True) & 
        (df["position"]==pos)
    ]
    return row.iloc[0]["prob"] if not row.empty else 0

def prob5SS(pos, df, method="annotated"):
    row = df[
        (df["type"]=="5prime") & 
        (df[method_column(method)]==True) & 
        (df["position"]==pos)
    ]
    return row.iloc[0]["prob"] if not row.empty else 0

# ====== Main Batch Processing ======
species_map = {
    'Arabidopsis': 'arabidopsis',
}

top_ks = [1000]

results = []
all_exon_scores = []

for top_k in top_ks:
    for code, name in species_map.items():
        print(f"Processing {code} with top_k = {top_k} for exon scores...")

        pattern = (
            f"Arabidopsis_ExonScore_k_1000/{name}_g_*.txt"
        )
        file_list = sorted(glob.glob(pattern))
        if not file_list:
            continue

        correct_exon_scores = []
        incorrect_exon_scores = []

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

            ann_pairs_list = (
                [(0, ann5ss[0])] +
                [(ann3ss[i], ann5ss[i+1]) for i in range(min(len(ann3ss), len(ann5ss)-1))] +
                [(ann3ss[-1], -1)]
            )
            sm_pairs_list = (
                [(0, sm5ss[0])] +
                [(sm3ss[i], sm5ss[i+1]) for i in range(min(len(sm3ss), len(sm5ss)-1))] +
                [(sm3ss[-1], -1)]
            )

            ann_pairs_set = set(ann_pairs_list)
            sm_pairs_set = set(sm_pairs_list)

            df_splice = parse_splice_file(txt_file)

            exon_table = {}
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if re.search(r"3SS,\s*5SS,\s*prob", line, re.IGNORECASE):
                    for subsequent_line in lines[i+1:]:
                        subsequent_line = subsequent_line.strip()
                        if not subsequent_line:
                            break
                        parts = subsequent_line.split(',')
                        if len(parts) >= 3:
                            try:
                                exon_3ss = int(parts[0].strip())
                                exon_5ss = int(parts[1].strip())
                                exon_prob = float(parts[2].strip())
                                exon_table[(exon_3ss, exon_5ss)] = exon_prob
                            except Exception:
                                continue
                    break

            for (three_site, five_site) in sm_pairs_set:
                p3 = prob3SS(three_site, df_splice, method="smsplice")
                p5 = prob5SS(five_site, df_splice, method="smsplice")


                if three_site == 0:
                    p3 = 1.0
                if five_site == -1:
                    p5 = 1.0

                if (three_site, five_site) in exon_table:
                    score = exon_table[(three_site, five_site)]
                else:
                    score = p3 * p5

                if score == 0.0:
                    continue

                is_correct = ((three_site, five_site) in ann_pairs_set)

                if is_correct:
                    correct_exon_scores.append(score)
                    all_exon_scores.append({
                        'top_k': top_k,
                        'species': code,
                        'label': 'correct',
                        'score': score
                    })
                else:
                    incorrect_exon_scores.append(score)
                    all_exon_scores.append({
                        'top_k': top_k,
                        'species': code,
                        'label': 'incorrect',
                        'score': score
                    })


        total_pred = len(correct_exon_scores) + len(incorrect_exon_scores)
        if total_pred == 0:
            precision_09 = None
            subset_09 = None
        else:
            preds_above_09 = [
                s for s in (correct_exon_scores + incorrect_exon_scores) 
                if s >= 0.9
            ]
            correct_above_09 = [s for s in correct_exon_scores if s >= 0.9]

            if len(preds_above_09) > 0:
                precision_09 = len(correct_above_09) / len(preds_above_09)
                subset_09 = len(preds_above_09) / total_pred
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
print("====== Exon Score Result ======")
print(df_result.to_string(index=False, float_format="{:.6f}".format))



