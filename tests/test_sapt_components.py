import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import extrapolator
import pandas as pd
import src
import numpy as np


@pytest.fixture
def all_sapt_terms():
    return [
        "SAPT ALPHA USER",
        "SAPT EXCHSCAL1",
        "SAPT EXCHSCAL3",
        "SAPT EXCHSCAL",
        "SAPT HF(2) ALPHA=0.0 ENERGY",
        # "SAPT HF(2),U ALPHA=0.0 ENERGY",
        # "SAPT HF(2),U ENERGY",
        "SAPT HF(2) ENERGY",
        "SAPT HF(3) ENERGY",
        "SAPT MP2(2) ENERGY",
        "SAPT MP2(3) ENERGY",
        "SAPT MP4 DISP",
        "SAPT CCD DISP",
        "SAPT0 ELST ENERGY",
        "SAPT0 EXCH ENERGY",
        "SAPT0 IND ENERGY",
        # "SAPT0 IND,U ENERGY",
        "SAPT0 DISP ENERGY",
        "SAPT0 TOTAL ENERGY",
        "SSAPT0 ELST ENERGY",
        "SSAPT0 EXCH ENERGY",
        "SSAPT0 IND ENERGY",
        # "SSAPT0 IND,U ENERGY",
        "SSAPT0 DISP ENERGY",
        "SSAPT0 TOTAL ENERGY",
        "SCS-SAPT0 ELST ENERGY",
        "SCS-SAPT0 EXCH ENERGY",
        "SCS-SAPT0 IND ENERGY",
        # "SCS-SAPT0 IND,U ENERGY",
        "SCS-SAPT0 DISP ENERGY",
        "SCS-SAPT0 TOTAL ENERGY",
        "SAPT2 ELST ENERGY",
        "SAPT2 EXCH ENERGY",
        "SAPT2 IND ENERGY",
        "SAPT2 DISP ENERGY",
        "SAPT2 TOTAL ENERGY",
        "SAPT2+ ELST ENERGY",
        "SAPT2+ EXCH ENERGY",
        "SAPT2+ IND ENERGY",
        "SAPT2+ DISP ENERGY",
        "SAPT2+ TOTAL ENERGY",
        "SAPT2+(CCD) ELST ENERGY",
        "SAPT2+(CCD) EXCH ENERGY",
        "SAPT2+(CCD) IND ENERGY",
        "SAPT2+(CCD) DISP ENERGY",
        "SAPT2+(CCD) TOTAL ENERGY",
        "SAPT2+DMP2 ELST ENERGY",
        "SAPT2+DMP2 EXCH ENERGY",
        "SAPT2+DMP2 IND ENERGY",
        "SAPT2+DMP2 DISP ENERGY",
        "SAPT2+DMP2 TOTAL ENERGY",
        "SAPT2+(CCD)DMP2 ELST ENERGY",
        "SAPT2+(CCD)DMP2 EXCH ENERGY",
        "SAPT2+(CCD)DMP2 IND ENERGY",
        "SAPT2+(CCD)DMP2 DISP ENERGY",
        "SAPT2+(CCD)DMP2 TOTAL ENERGY",
        "SAPT2+(3) ELST ENERGY",
        "SAPT2+(3) EXCH ENERGY",
        "SAPT2+(3) IND ENERGY",
        "SAPT2+(3) DISP ENERGY",
        "SAPT2+(3) TOTAL ENERGY",
        "SAPT2+(3)(CCD) ELST ENERGY",
        "SAPT2+(3)(CCD) EXCH ENERGY",
        "SAPT2+(3)(CCD) IND ENERGY",
        "SAPT2+(3)(CCD) DISP ENERGY",
        "SAPT2+(3)(CCD) TOTAL ENERGY",
        "SAPT2+(3)DMP2 ELST ENERGY",
        "SAPT2+(3)DMP2 EXCH ENERGY",
        "SAPT2+(3)DMP2 IND ENERGY",
        "SAPT2+(3)DMP2 DISP ENERGY",
        "SAPT2+(3)DMP2 TOTAL ENERGY",
        "SAPT2+(3)(CCD)DMP2 ELST ENERGY",
        "SAPT2+(3)(CCD)DMP2 EXCH ENERGY",
        "SAPT2+(3)(CCD)DMP2 IND ENERGY",
        "SAPT2+(3)(CCD)DMP2 DISP ENERGY",
        "SAPT2+(3)(CCD)DMP2 TOTAL ENERGY",
        "SAPT2+3 ELST ENERGY",
        "SAPT2+3 EXCH ENERGY",
        "SAPT2+3 IND ENERGY",
        "SAPT2+3 DISP ENERGY",
        "SAPT2+3 TOTAL ENERGY",
        "SAPT2+3(CCD) ELST ENERGY",
        "SAPT2+3(CCD) EXCH ENERGY",
        "SAPT2+3(CCD) IND ENERGY",
        "SAPT2+3(CCD) DISP ENERGY",
        "SAPT2+3(CCD) TOTAL ENERGY",
        "SAPT2+3DMP2 ELST ENERGY",
        "SAPT2+3DMP2 EXCH ENERGY",
        "SAPT2+3DMP2 IND ENERGY",
        "SAPT2+3DMP2 DISP ENERGY",
        "SAPT2+3DMP2 TOTAL ENERGY",
        "SAPT2+3(CCD)DMP2 ELST ENERGY",
        "SAPT2+3(CCD)DMP2 EXCH ENERGY",
        "SAPT2+3(CCD)DMP2 IND ENERGY",
        "SAPT2+3(CCD)DMP2 DISP ENERGY",
        "SAPT2+3(CCD)DMP2 TOTAL ENERGY",
    ]

def test_extrapolate_energies_df(all_sapt_terms):
    df = pd.read_pickle("hbc6-plat-atz-all.pkl")
    starting_cols = [
        "SAPT EXCHSCAL1",
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
        "SAPT ELST10,R ENERGY",
        "SAPT EXCH10 ENERGY",
        "SAPT EXCH10(S^2) ENERGY",
        "SAPT IND20,R ENERGY",
        "SAPT EXCH-IND20,R ENERGY",
        # "SAPT IND20,U ENERGY",
        # "SAPT EXCH-IND20,U ENERGY",
    ]
    df.columns = df.columns.values + " (TZ)"
    print(df.columns.values)
    for col in starting_cols:
        s = col + " (TZ)"
        df[col] = df[s]
    df = src.extrap_df.compute_sapt_terms(df)
    print(df.columns.values)
    for i in all_sapt_terms:
        if i in df.columns:
            target = i + " (TZ)"
            print(i, target)
            assert np.all(df[target] - df[i] < 1e-12)





