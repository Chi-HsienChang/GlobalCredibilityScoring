# Global Credibility Scoring (GCS): Source Code and Arabidopsis Example

1. Download the pre-trained model model_Arabidopsis.pkl from the private Figshare link below:
https://figshare.com/s/d7378e5b67a420dc8322 
(Alternatively, users can download the model from the official SMsplice GitHub repository https://github.com/kmccue/smsplice, but they will need to manually pass the parameters to the corresponding function call of GCS method.)

2. Install awkde from source:
>>> git clone https://github.com/mennthor/awkde
>>> pip install ./awkde

3. Install Dependencies
>>> pip install numpy pandas cython scikit-learn

4. Compile the GCS.pyx file using the following command:
>>> python3 setup.py build_ext --inplace

5. Run the following commands to execute GCS with k = 100 for genes 0 in the Arabidopsis dataset:
>>> python3 run_Arabidopsis.py --k 100 --gene_index 0

6. After execution, a new folder named Example will be created.
This folder contains three subdirectories — Exon_Score, Intron_Score, and SS_Score — each storing the corresponding output files for the selected genes.


7. Example Output (After running finishes, the output will look like this:)

Dataset = Arabidopsis
Gene = AT2G01100 (Index 0)
Length = 1778
Running with Top_k = 100

################ Computational Time ################
[Exon Score] completed in 0.38 seconds.
[Intron Score] completed in 0.38 seconds.
[Splice Site Scoring] completed in 0.39 seconds.

#################### Results #######################
All score files have been successfully saved to:
 [1] ./Example/Exon_Score
 [2] ./Example/Intron_Score
 [3] ./Example/SS_Score
####################################################
