import os
import sys
import pickle
import numpy as np
import pandas as pd
import argparse
import time
import GCS

# Create an argument parser
parser = argparse.ArgumentParser(description="Run SMsplice GCS scoring on a specified gene.")

# Add arguments
parser.add_argument("--k", type=int, required=True, help="Number of top parses to keep (Top_k)")
parser.add_argument("--gene_index", type=int, required=True, help="Index of the gene to process")

# Parse the arguments
args = parser.parse_args()
Top_k = args.k
Gene_Index = args.gene_index
GENE_INDICES = [Gene_Index]

# Checkpoint interval: larger values lower memory usage
CHECKPOINT_INTERVAL = 1

# Example species/model identifier
SPECIES = 'Arabidopsis'
# Base folder for output examples
OUTPUT_BASE = './Example'

# -----------------------------------------------------------------------------
# Utility: Load pickled model and predictions
# -----------------------------------------------------------------------------
def load_model_data(species):
    """
    Load the SMsplice model parameters from a pickle file for the given species.

    Returns:
        dict: A dictionary containing model parameters and sequence data.
    """
    filename = f'model_{species}.pkl'
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_prediction_data(species):
    """
    Load the SMsplice prediction outputs from a pickle file for the given species.

    Returns:
        dict: A dictionary containing prediction results (true and predicted splice sites).
    """
    filename = f'SMsplice_predictions_{species}.pkl'
    with open(filename, 'rb') as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Functions to compute and save scores
# -----------------------------------------------------------------------------
def write_exon_scores(i, data, model, species):
    """
    Compute and save exon confidence scores for sample index i.
    The scores are computed via forward-backward on exon parses.
    """
    gene_id = model['testGenes'][i]
    length = model['lengths'][i]
    out_dir = os.path.join(OUTPUT_BASE, 'Exon_Score')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{species}_Gene_{gene_id}_Index_{i}.txt')

    # Redirect stdout to output file
    original_stdout = sys.stdout
    sys.stdout = open(out_file, 'w')

    print(f"Gene = {gene_id}")
    print(f"Index = {i}")
    print(f"Length = {length}")
    print()

    # True annotated splice sites
    true5 = data['trueFives_all'][i]
    true3 = data['trueThrees_all'][i]
    print(f"Annotated 5'SS: {true5}")
    print(f"Annotated 3'SS: {true3}")

    # SMsplice-predicted splice sites
    pred5 = data['predFives_all'][i]
    pred3 = data['predThrees_all'][i]
    print(f"SMsplice 5'SS: {pred5}")
    print(f"SMsplice 3'SS: {pred3}")
    print()

    # Run global credibility scoring for exons (pass all args positionally)
    first_dict, middle_dict, logZ, F_curr, B_curr = GCS.GCS_Exon(
        model['sequences'][i],
        model['pME'], model['pELF'], model['pIL'], model['pEE'],
        model['pELM'], model['pEO'], model['pELL'],
        model['emissions5'][i], model['emissions3'][i],
        length, CHECKPOINT_INTERVAL, Top_k
    )
    last_dict, _ = GCS.forward_backward_last_exon(
        F_curr, B_curr, length, logZ
    )

    # Aggregate all exon scores
    all_exons = list(first_dict.items()) + list(middle_dict.items()) + list(last_dict.items())

    print(f"Partition function (log Z): {logZ}")
    print(f"#Exons = {len(all_exons)}")
    print("\n3'SS, 5'SS, Exon Score")
    for (a, b), score in sorted(all_exons, key=lambda x: x[1], reverse=True):
        print(f"{a}, {b-1}, {score}")

    # Restore stdout
    sys.stdout.close()
    sys.stdout = original_stdout


def write_intron_scores(i, data, model, species):
    """
    Compute and save intron confidence scores for sample index i.
    """
    gene_id = model['testGenes'][i]
    length = model['lengths'][i]
    out_dir = os.path.join(OUTPUT_BASE, 'Intron_Score')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{species}_Gene_{gene_id}_Index_{i}.txt')

    original_stdout = sys.stdout
    sys.stdout = open(out_file, 'w')

    print(f"Gene = {gene_id}")
    print(f"Index = {i}")
    print(f"Length = {length}")
    print()

    print(f"Annotated 5'SS: {data['trueFives_all'][i]}")
    print(f"Annotated 3'SS: {data['trueThrees_all'][i]}")
    print(f"SMsplice 5'SS: {data['predFives_all'][i]}")
    print(f"SMsplice 3'SS: {data['predThrees_all'][i]}")
    print()

    # Run global credibility scoring for introns
    intron_dict, logZ = GCS.GCS_Intron(
        model['sequences'][i],
        model['pME'], model['pELF'], model['pIL'], model['pEE'],
        model['pELM'], model['pEO'], model['pELL'],
        model['emissions5'][i], model['emissions3'][i],
        length, CHECKPOINT_INTERVAL, Top_k
    )

    print(f"Partition function (log Z): {logZ}")
    print(f"#Introns = {len(intron_dict)}")
    print("\n5'SS, 3'SS, Intron Score")
    for (a, b), score in sorted(intron_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{a}, {b-1}, {score}")

    sys.stdout.close()
    sys.stdout = original_stdout


def write_ss_scores(i, data, model, species):
    """
    Compute and save individual splice-site confidence scores (5' and 3').
    """
    gene_id = model['testGenes'][i]
    length = model['lengths'][i]
    out_dir = os.path.join(OUTPUT_BASE, 'SS_Score')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{species}_Gene_{gene_id}_Index_{i}.txt')

    original_stdout = sys.stdout
    sys.stdout = open(out_file, 'w')

    print(f"Gene = {gene_id}")
    print(f"Index = {i}")
    print(f"Length = {length}")
    print()
    print(f"Annotated 5'SS: {data['trueFives_all'][i]}")
    print(f"Annotated 3'SS: {data['trueThrees_all'][i]}")
    print(f"SMsplice 5'SS: {data['predFives_all'][i]}")
    print(f"SMsplice 3'SS: {data['predThrees_all'][i]}")
    print()

    # Run global credibility scoring for splice sites
    posterior, logZ = GCS.GCS_SS(
        model['sequences'][i],
        model['pME'], model['pELF'], model['pIL'], model['pEE'],
        model['pELM'], model['pEO'], model['pELL'],
        model['emissions5'][i], model['emissions3'][i],
        length, CHECKPOINT_INTERVAL, Top_k
    )

    print(f"Partition function (log Z): {logZ}")
    print("\nSorted 5'SS Scores:")
    five_scores = [(pos-1, posterior[pos][5]) for pos in range(1, length) if 5 in posterior[pos]]
    for pos, score in sorted(five_scores, key=lambda x: x[1], reverse=True):
        print(f"Position {pos}: {score}")

    print("\nSorted 3'SS Scores:")
    three_scores = [(pos-1, posterior[pos][3]) for pos in range(1, length) if 3 in posterior[pos]]
    for pos, score in sorted(three_scores, key=lambda x: x[1], reverse=True):
        print(f"Position {pos}: {score}")

    sys.stdout.close()
    sys.stdout = original_stdout

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Load model parameters and sequence data
    model_data = load_model_data(SPECIES)
    # Load SMsplice predictions
    prediction_data = load_prediction_data(SPECIES)

    # Iterate over chosen genes
    for idx in GENE_INDICES:
        gene_id = model_data['testGenes'][idx]
        length = model_data['lengths'][idx]
        print(f"Dataset = {SPECIES}")
        print(f"Gene = {gene_id} (Index {idx})")
        print(f"Length = {length}")
        print(f"Running with Top_k = {Top_k}")

        print("\n################ Computational Time ################")
        # Time tracking
        start_exon = time.time()
        write_exon_scores(idx, prediction_data, model_data, SPECIES)
        end_exon = time.time()
        print(f"[Exon Score] completed in {end_exon - start_exon:.2f} seconds.")

        start_intron = time.time()
        write_intron_scores(idx, prediction_data, model_data, SPECIES)
        end_intron = time.time()
        print(f"[Intron Score] completed in {end_intron - start_intron:.2f} seconds.")

        start_ss = time.time()
        write_ss_scores(idx, prediction_data, model_data, SPECIES)
        end_ss = time.time()
        print(f"[Splice Site Scoring] completed in {end_ss - start_ss:.2f} seconds.")

    print("\n#################### Results #######################")
    print(f"All score files have been successfully saved to:\n"
        f" [1] {os.path.join(OUTPUT_BASE, 'Exon_Score')}\n"
        f" [2] {os.path.join(OUTPUT_BASE, 'Intron_Score')}\n"
        f" [3] {os.path.join(OUTPUT_BASE, 'SS_Score')}")
    print("####################################################")

