import pandas as pd
from glob import glob
from pprint import pprint as pp
import os


def extrapolate_energies(C1, C2, E1, E2):
    """Test extrapolation of two individual correlation energies, to
    help test the operations on the dataframe

    Arguments:
        C1: Cardinal number of basis set 1
        C2: Cardinal number of basis set 2
        E1: Energy in basis set 1
        E2: Energy in basis set 2

    Returns:
        Extrapolated energy

    Note:
        Always be careful about mixing integer and floating point
        arithmetic.  It looks like Python converts stuff for us correctly
        in this case.
    """

    # Set "large" basis (L) and "small" basis (S) cardinal numbers and energies
    # (the order matters)
    if C1 == C2 + 1:
        L = C1
        S = C2
        EL = E1
        ES = E2
    elif C2 == C1 + 1:
        L = C2
        S = C1
        EL = E2
        ES = E1
    else:
        raise ValueError("Cardinal numbers must be adjacent integers")

    E_CBS = (L**3 * EL - S**3 * ES) / (L**3 - S**3)

    return E_CBS


def extrapolate_energies_df(f1, f2, df_out):
    # read a double-zeta file
    df_d = pd.read_pickle("sapt_ref_data/adz/hbc6-plat-adz-all.pkl")
    # read a triple-zeta file
    df_t = pd.read_pickle("sapt_ref_data/atz/hbc6-plat-atz-all.pkl")
    cols = df_d.columns.values

    # now extrapolate the SAPT0 dispersion energy
    df_tmp = (3**3) / (3**3 - 2**3) * df_t["SAPT0 DISP ENERGY"] - (
        (2**3) / (3**3 - 2**3)
    ) * df_d["SAPT0 DISP ENERGY"]
    # Need to grab all columns that are 'correlated' terms

    extrap_columns = [
        "SAPT DISP20 ENERGY",
        "SAPT EXCH-DISP20 ENERGY",
    ]

    df_d.columns = df_d.columns.values + " (DZ)"
    df_t.columns = df_t.columns.values + " (TZ)"
    df = pd.concat([df_d, df_t], axis=1)
    for i in extrap_columns:
        df[i] = df.apply(
            lambda r: extrapolate_energies(2, 3, r[i + " (DZ)"], r[i + " (TZ)"]), axis=1
        )

    df["SAPT0 DISP ENERGY"] = (
        df["SAPT DISP20 ENERGY (TZ)"] + df["SAPT EXCH-DISP20 ENERGY (TZ)"]
    )
    df["SAPT0 ELST ENERGY"] = df["SAPT ELST10,R ENERGY (TZ)"]
    df["SAPT0 IND ENERGY"] = (
        # NOTE: SAPT EXCHSCAL is 1 due to alpha=0.0, 2023-07-26
        df["SAPT IND20,R ENERGY (TZ)"] * df['SAPT EXCHSCAL (TZ)']
        + df["SAPT HF(2) ENERGY (TZ)"]
        + df["SAPT EXCH-IND20,R ENERGY (TZ)"]
    )
    df["SAPT0 EXCH ENERGY"] = df["SAPT EXCH10 ENERGY (TZ)"]
    df["SAPT0 TOTAL ENERGY"] = (
        df["SAPT0 DISP ENERGY"]
        + df["SAPT0 ELST ENERGY"]
        + df["SAPT0 IND ENERGY"]
        + df["SAPT0 EXCH ENERGY"]
    )
    for n, r in df.iterrows():
        disp = r["SAPT0 DISP ENERGY"]
        exch = r["SAPT0 EXCH ENERGY"]
        exch_tz = r["SAPT0 EXCH ENERGY (TZ)"]
        ind = r["SAPT0 IND ENERGY"]
        ind_tz = r["SAPT0 IND ENERGY (TZ)"]
        elst = r["SAPT0 ELST ENERGY"]
        elst_tz = r["SAPT0 ELST ENERGY (TZ)"]
        tot = r["SAPT0 TOTAL ENERGY"]
        sapt_alpha = r['SAPT ALPHA (TZ)']
        assert abs(elst - elst_tz) < 1e-12
        assert abs(ind - ind_tz) < 1e-12
        assert abs(exch - exch_tz) < 1e-12

    tz_columns = [
        "SAPT ELST10,R ENERGY",
        "SAPT EXCH10 ENERGY",
        "SAPT EXCH10(S^2) ENERGY",
        "SAPT IND20,R ENERGY",
        "SAPT EXCH-IND20,R ENERGY",
        "SAPT HF(2) ALPHA=0.0 ENERGY",
        "SAPT HF(2) ENERGY",
        "SAPT HF(3) ENERGY",
    ]
    for i in tz_columns:
        df[i] = df[i + " (TZ)"]

    subset = [i for i in df.columns.values if "(TZ)" not in i and "(DZ)" not in i]
    print(subset)
    df_subset = df[subset]
    df_subset.to_pickle(df_out)

    # I need to store that as a series inside a dataframe df_dt ... if I
    # initialize it as empty first can I compute it directly to the desired
    # location?  Or else I need to copy the computed series into the dataframe
    return

def generate_output_pkls():
    # Ok mess around for now and test
    fs_adz = glob("sapt_ref_data/adz/*.pkl")
    fs_atz = glob("sapt_ref_data/atz/*.pkl")
    print(fs_adz)
    print(fs_atz)
    for i in fs_atz:
        db_atz = i.split("/")[-1].split("-")[0]
        for j in fs_adz:
            db_adz = j.split("/")[-1].split("-")[0]
            if db_atz == db_adz:
                print(db_atz, db_adz)
                print(i, j)
                df_path_out = i.replace("atz", "adtz")
                print(df_path_out)
                dir_path = "/".join(df_path_out.split("/")[:-1])
                print(dir_path)
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                extrapolate_energies_df(i, j, df_path_out)
                return


def main():
    generate_output_pkls()
    # test output
    df = pd.read_pickle("sapt_ref_data/adtz/hbc6-plat-adtz-all.pkl")
    print(df.columns)
    return


if __name__ == "__main__":
    main()
