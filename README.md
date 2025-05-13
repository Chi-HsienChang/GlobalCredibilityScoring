# Global Credibility Scoring (GCS)

Each folder contains a corresponding README file with instructions for execution.

## Source_code
   Contains the core implementation of the GCS algorithm.  
   - `GCS.pyx` implements the main scoring routine in Cython.  
   - `setup.py` is used to compile the Cython code.  
   - `run_Arabidopsis.py` is an example script that runs GCS on a specific gene from the Arabidopsis dataset.  
   - The example uses a pre-trained model (`model_Arabidopsis.pkl`) and saves the resulting exon, intron, and splice site scores in the `Example/` directory.
   - Due to the 100MB file size limit, only the Arabidopsis results are provided here. Results for other species can be downloaded via the following anonymous Figshare link:  
     - Human: https://figshare.com/s/cba533b5ae4879e263de
     - Mouse: https://figshare.com/s/402f5d01b280e1a7e910
     - Arabidopsis: https://figshare.com/s/34134aabbb1a0cf70ceb
     - Zebrafish: https://figshare.com/s/b1ea6c0cdba2241552c2
     - Moth: https://figshare.com/s/a3cbe1578e2ce1eeb09b
     - Fly: https://figshare.com/s/f9f410b44a43253507e6

## Precision_results_with_k_1000 
   Contains evaluation scripts and output tables for exon, intron, and splice site precision on the Arabidopsis dataset using top-k parsing (k = 1000).  
   - `Arabidopsis_ExonScore_k_1000/`: Stores per-gene exon-level credibility scores.  
   - `Arabidopsis_IntronScore_k_1000/`: Stores per-gene intron-level credibility scores.  
   - `Arabidopsis_SpliceSiteScore_k_1000/`: Stores per-gene splice site credibility scores.  
   - Scripts such as `Exon_precision.py`, `Intron_precision.py`, and `SpliceSite_precision.py` compute precision metrics across different confidence thresholds and reproduce results reported in the paper.
   - Due to the 100MB file size limit, only the Arabidopsis results are provided here. Results for other species can be downloaded via the following anonymous Figshare link: https://figshare.com/s/43a8e3f66dd8f02eef23

## AS_results_with_k_1000
   Contains the evaluation of Global Credibility Scoring (GCS) on alternative splicing (AS) events using top-k parsing (k = 1000) across three species: human, mouse, and Arabidopsis.  
   - Includes subfolders for four types of AS events: SE (Skipped Exon), RI (Retained Intron), A5SS (Alternative 5' Splice Site), and A3SS (Alternative 3' Splice Site).  
   - Each subfolder contains species-specific scripts (e.g., `SE_Human_k_1000.py`) for plotting the distribution of GCS scores across AS event types.  
   - Required data (GCS scores and AS annotations) are available via Figshare and should be placed in the same directory.  
   - Output figures are saved as `SE_Human.png`, `RI_Mouse.png`, `A5SS_Arabidopsis.png`, etc., summarizing GCS score distributions in AS versus constitutive regions.