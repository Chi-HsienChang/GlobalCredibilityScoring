# Evaluation of Global Credibility Scoring (GCS) on Alternative Splice Events (k = 1000)

1. Download the GSC scores of Human, Mouse, and Arabidopsis from the private Figshare link below: 
https://figshare.com/s/d30e97dfc1e9db4d3dfb

The downloaded folder is placed in ./AS_results_with_k_1000

2. Download the AS data of Human, Mouse, and Arabidopsis from the private Figshare link below: 
https://figshare.com/s/7889dbbc79a92d12cab5

The downloaded folder is placed in ./AS_results_with_k_1000

3. The corresponding AS event analysis can now be performed using the relevant folder.

Install:
>>> python3 -m pip install --upgrade pip
>>> python3 -m pip install numpy pandas matplotlib seaborn scipy

## Skipped Exon (SE)

Figures will be saved as SE_Human.png, SE_Mouse.png, and SE_Arabidopsis.png

>>> cd SE

- Human
>>> python3 SE_Human_k_1000.py
- Mouse
>>> python3 SE_Mouse_k_1000.py
- Arabidopsis
>>> python3 SE_Arabidopsis_k_1000.py


## Retained Intron (RI)

Figures will be saved as RI_Human.png, RI_Mouse.png, and RI_Arabidopsis.png

>>> cd RI

- Human
>>> python3 RI_Human_k_1000.py
- Mouse
>>> python3 RI_Mouse_k_1000.py
- Arabidopsis
>>> python3 RI_Arabidopsis_k_1000.py


## Alternative 5' Splice Site (A5SS)

Figures will be saved as A5SS_Human.png, A5SS_Mouse.png, and A5SS_Arabidopsis.png.

>>> cd A5SS

- Human
>>> python3 A5SS_Human_k_1000.py
- Mouse
>>> python3 A5SS_Mouse_k_1000.py
- Arabidopsis
>>> python3 A5SS_Arabidopsis_k_1000.py



## Alternative 3' Splice Site (A3SS)

Figures will be saved as A3SS_Human.png, A3SS_Mouse.png, and A3SS_Arabidopsis.png

>>> cd A3SS

- Human
>>> python3 A3SS_Human_k_1000.py
- Mouse
>>> python3 A3SS_Mouse_k_1000.py
- Arabidopsis
>>> python3 A3SS_Arabidopsis_k_1000.py

