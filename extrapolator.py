import pandas as pd
from glob import glob
from pprint import pprint as pp
import os
import src

# TODO: add path to psi4/db to PYTHONPATH to get access to database
# ex. import BBI
#     BBI.bind[key]

# might look at cdsgroup/qcdb --branch data
#   has citations for binds based on references


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


def extrapolate_energies_df(
    c1_data: dict = {
        "df_path": "sapt_ref_data/adz/hbc6-plat-adz-all.pkl",
        "c_label": "DZ",
        "C1": 2,
    },
    c2_data: dict = {
        "df_path": "sapt_ref_data/atz/hbc6-plat-atz-all.pkl",
        "c_label": "TZ",
        "C2": 3,
    },
    df_out: str = "sapt_ref_data/cbs/hbc6-plat-adtz-all.pkl",
):
    """Take in two dictionaries, one for results in a smaller basis
    (c1_data), and one for results in a larger basis (c2_data), and
    a path for an output dataframe in pkl format.  The dictionaries
    contain a path for a dataframe in pkl format, a label (e.g.,
    DZ, TZ), and a cardinal number for the basis set (e.g., 2 for DZ,
    3 for TZ, etc.).  The cardinal numbers must be adjacent.

    The function then does a 2-point Helgaker extrapolation on columns
    with SAPT data matching one of the labels below in extrap_columns
    (only electron-correlation related columns).  It copies over
    Hartree-Fock level columns (e.g., SAPT ELST10,R ENERGY and others
    listed in copy_from_large_basis_columns).

    The resulting dataframe is written out to the pkl file specified

    A couple of default dictionaries are provided for testing purposes.
    """

    # some notes about exchange-scaling:
    # many of the fundamental SAPT terms are computed using the
    # single-exchange (S^2) approximation.  In Ed Hohenstein's original
    # SAPT code, we scaled all of these by the ratio
    # E_{exch}^{(10)} / E_{exch}^{(10)(S^2)} = "SAPT EXCHSCAL1"
    # to approximately compensate for the S^2 approximation.  However,
    # I do not recall any theory paper that really demonstrated that
    # this scaling is necessarily very effective (in SAPT(DFT) Hesselmann
    # found that scaling E_{exch-disp}^{(20)} by a simple constant
    # was more effective than using the above ratio.
    #
    # Ultimately, we decided at some point to reverse Ed's exchange
    # scaling and not scale these terms, especially for higher-order
    # SAPT.  I don't recall if other codes commonly scale
    # E_{exch-disp}^{(20)} and E_{exch-ind}^{(20)} or not, but even
    # if there is some literature precedent for that, I got concerned
    # about scaling the higher-order (S^2) terms without more evidence
    # that it was a good idea.  We universally stopped scaling any of
    # these terms (even for SAPT0) by default, although the scaling
    # can be turned back on, which would then default to using
    # the above ratio, SAPT EXCHSCAL1, or alternatively this factor
    # can also be used raised to higher powers, determined by
    # "SAPT ALPHA", providing the final scaling factor "SAPT EXCHSCAL".
    #
    # If no exchange scaling is done, then that is effectively the same
    # as using SAPT ALPHA = 0, yielding SAPT EXCHSCAL = 1.0.
    #
    # A special value of SAPT ALPHA = 3 is used to obtain the
    # so-called scaled SAPT0, or sSAPT0, which we found to work better
    # for many short-range contacts (e.g., 10.1039/c8cp02029a).
    # (On the other hand, we found that some *very* close contacts
    # get over-corrected by sSAPT0, e.g., the Splinter dataset paper 2023).
    # The exchange scaling ratio to the third power is stored in
    # SAPT EXCHSCAL3, which is used the formulas to compute sSAPT energies.
    # -CDS 7/27/23

    df_c1 = pd.read_pickle(c1_data["df_path"])
    df_c2 = pd.read_pickle(c2_data["df_path"])
    C1 = c1_data["C1"]
    C2 = c2_data["C2"]
    c1_label = c1_data["c_label"]
    c2_label = c2_data["c_label"]

    # now extrapolate the SAPT0 dispersion energy
    # df_tmp = (3**3) / (3**3 - 2**3) * df_t["SAPT0 DISP ENERGY"] - (
    #     (2**3) / (3**3 - 2**3)
    # ) * df_d["SAPT0 DISP ENERGY"]
    # Need to grab all columns that are 'correlated' terms

    # list all possible columns to extrapolate.  if a lower level of
    # SAPT was computed, some of these columns might not be available
    # do not extrapolate quantities that are derived from more fundamental
    # quantities ... just re-compute those using our SAPT variable
    # computation code adapted from Psi4
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
        "SAPT MP2 CORRELATION ENERGY",  # supermolecular MP2 E_corr for dMP2
    ]
    copy_from_larger_basis_columns = [
        "SAPT ALPHA",
        "SAPT ELST10,R ENERGY",
        "SAPT EXCH10 ENERGY",
        "SAPT EXCH10(S^2) ENERGY",
        "SAPT IND20,R ENERGY",
        "SAPT EXCH-IND20,R ENERGY",
        "SAPT IND20,U ENERGY",
        "SAPT EXCH-IND20,U ENERGY",
        "SAPT HF TOTAL ENERGY",
    ]

    df_c1.columns = df_c1.columns.values + f" ({c1_label})"
    df_c2.columns = df_c2.columns.values + f" ({c2_label})"
    df = pd.concat([df_c1, df_c2], axis=1)

    # copy HF-level data (not depending on electron correlation) from the
    # larger basis, just like we would do in focal-point methods
    for i in copy_from_larger_basis_columns:
        if f"{i} ({c2_label})" in df_c2.columns.values:
            df[i] = df[i + f" ({c2_label})"]

    # extrapolate the electron-correlation dependent terms
    for i in extrap_columns:
        df[i] = df.apply(
            lambda r: extrapolate_energies(
                2, 3, r[i + f" ({c1_label})"], r[i + f" ({c2_label})"]
            ),
            axis=1,
        )
    subset = [
        i
        for i in df.columns.values
        if f" ({c1_label})" not in i and f" ({c2_label})" not in i
    ]
    df = df[subset].copy()
    # now compute SAPT terms from the extrapolated energies
    df = src.compute_sapt_terms.compute_sapt_terms(df)
    print(len(df))
    # now select the new data for export
    df.to_pickle(df_out)
    return


def generate_output_pkls():
    # Ok mess around for now and test
    fs_adz = glob("sapt_ref_data/adz/*.pkl")
    fs_atz = glob("sapt_ref_data/atz/*.pkl")
    fs_qz = glob("sapt_ref_data/aqz/*.pkl")
    print("Creating: aDZ + aTZ -> aDTZ")
    for i in fs_atz:
        db_atz = i.split("/")[-1].split("-")[0]
        for j in fs_adz:
            db_adz = j.split("/")[-1].split("-")[0]
            if db_atz == db_adz:
                df_path_out = i.replace("atz", "adtz")
                print(db_atz, i, j)
                print("Extrapolated results:", df_path_out, end="\n\n")
                dir_path = "/".join(df_path_out.split("/")[:-1])
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                extrapolate_energies_df(
                    c1_data={
                        "df_path": j,
                        "c_label": "DZ",
                        "C1": 2,
                    },
                    c2_data={
                        "df_path": i,
                        "c_label": "TZ",
                        "C2": 3,
                    },
                    df_out=df_path_out,
                )
    print("Creating: aTZ + aQZ -> aTQZ")
    for i in fs_atz:
        db_atz = i.split("/")[-1].split("-")[0]
        for j in fs_qz:
            db_aqz = j.split("/")[-1].split("-")[0]
            if db_atz == db_aqz:
                print(db_atz, i, j)
                print("Extrapolated results:", df_path_out, end="\n\n")
                df_path_out = i.replace("atz", "atqz")
                dir_path = "/".join(df_path_out.split("/")[:-1])
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                extrapolate_energies_df(
                    c1_data={
                        "df_path": i,
                        "c_label": "TZ",
                        "C1": 3,
                    },
                    c2_data={
                        "df_path": j,
                        "c_label": "QZ",
                        "C2": 4,
                    },
                    df_out=df_path_out,
                )
    return


def main():
    generate_output_pkls()
    # example reading output
    # df = pd.read_pickle("sapt_ref_data/adtz/hbc6-plat-adtz-all.pkl")
    # df = pd.read_pickle("sapt_ref_data/atqz/hbc6-plat-atqz-all.pkl")
    # print(df.columns.values)
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    # print(df)
    return


if __name__ == "__main__":
    main()
