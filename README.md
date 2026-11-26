# DS3006 Quantitative Finance: Group Assignment 2 (Q1, Q3, Q5)

| File | What it does |
|---|---|
| `assignment2_final.py` | All computations. Writes every table to `tables/` and every figure to `figs/`. |
| `assignment2.ipynb` | The same code as a notebook (run all cells). |
| `requirements.txt` | Python packages, with the versions we tested. |

## Data
Download the NSE F&O UDiFF Common Bhavcopy for 6 Nov 2026
(`BhavCopy_NSE_FO_0_0_0_20261106_F_0000.csv.zip`, nseindia.com > All Reports > Derivatives)
and put the .csv or the .zip in `data/`.

## Run
    pip install -r requirements.txt
    python assignment2_final.py      # or run assignment2.ipynb
    python build_report.py

The Q1 random seed is 2024. Strikes, rate and the other inputs are in the CONFIG block at the top of `assignment2_final.py`.
