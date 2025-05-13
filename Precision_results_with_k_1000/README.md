# Global Credibility Scoring (GCS) Results (k = 1000)

This directory contains the Global Credibility Scoring (k = 1000) results on the Arabidopsis dataset. 
The results cover 1117 genes and are organized into three folders:

- Arabidopsis_ExonScore_k_1000/
- Arabidopsis_IntronScore_k_1000/
- Arabidopsis_SpliceSiteScore_k_1000/

It can reproduce the precision results reported in the paper using the following commands:

>>> python3 Exon_precision.py 

====== Exon Score Result ======
 top_k     species  precision with >=0.9  (Subset%)
  1000 Arabidopsis              0.957605  34.645669


>>> python3 Intron_precision.py 

====== Intron Score Result ======
 top_k     species  precision with >=0.9  (Subset%)
  1000 Arabidopsis              0.962588  36.022582


>>> python3 SpliceSite_precision.py

====== Splice Site Score Result ======
species = Arabidopsis
Top_k   = 1000
precision:
[0.9, 1.0]  (Subset%)
0.971321  56.750267
[0.8,0.9)  (Subset%)
0.896071  14.034152
[0.7,0.8)  (Subset%)
0.784000  8.893632
[0.6,0.7)  (Subset%)
0.668428  6.732480
[0.5,0.6)  (Subset%)
0.563312  5.478477
[0.4,0.5)  (Subset%)
0.452381  4.108858