import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import extrapolator
import pandas as pd
import src
import numpy as np

def test_extrapolate_energies_df():
    df = pd.read_pickle("hbc6-plat-atz-all.pkl")
    extrap_columns = [
        "SAPT DISP20 ENERGY",
        "SAPT DISP21 ENERGY",
        "SAPT DISP22(S)(CCD) ENERGY",
        "SAPT DISP22(SDQ) ENERGY",
        "SAPT DISP22(T) ENERGY",
        "SAPT DISP22(T)(CCD) ENERGY",
        "SAPT DISP30 ENERGY",
        "SAPT ELST12,R ENERGY",
        "SAPT ELST13,R ENERGY",
        "SAPT EST.DISP22(T) ENERGY",
        "SAPT EST.DISP22(T)(CCD) ENERGY",
        "SAPT EXCH-DISP30 ENERGY",
        "SAPT EXCH-IND-DISP30 ENERGY",
        "SAPT EXCH-IND22 ENERGY",
        "SAPT EXCH-IND30,R ENERGY",
        "SAPT EXCH11(S^2) ENERGY",
        "SAPT EXCH12(S^2) ENERGY",
        "SAPT IND-DISP30 ENERGY",
        "SAPT IND22 ENERGY",
        "SAPT IND30,R ENERGY",
        "SAPT MP2 CORRELATION ENERGY",
    ]
    for col in extrap_columns:
        target = col + " (TZ)"
        df[target] = df[col]
    df = src.extrap_df.compute_sapt_terms(df)
    print(df[extrap_columns[0]])
    for col in df.columns.values:
        if col.endswith("ENERGY"):
            target = col + " (TZ)"
            assert np.all(df[target] - df[col] < 1e-12)

