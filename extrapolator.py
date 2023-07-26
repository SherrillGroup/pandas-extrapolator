import pandas as pd
from glob import glob


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


def main():

    # Ok mess around for now and test

    # files = glob("sapt_ref_data/adz/*.pkl")

    # read a double-zeta file
    df_d = pd.read_pickle("sapt_ref_data/adz/hbc6-plat-adz-all.pkl")
    # read a triple-zeta file
    df_t = pd.read_pickle("sapt_ref_data/atz/hbc6-plat-atz-all.pkl")

    # now extrapolate the SAPT0 dispersion energy
    df_tmp = (3**3) / (3**3 - 2**3) * df_t["SAPT0 DISP ENERGY"] - (
        (2**3) / (3**3 - 2**3)
    ) * df_d["SAPT0 DISP ENERGY"]
    # Need to grab all columns that are 'correlated' terms

    extrap_columns = [
        "SAPT0 DISP ENERGY",
    ]


    df_d.columns = df_d.columns.values + " (DZ)"
    df_t.columns = df_t.columns.values + " (TZ)"
    df = pd.concat([df_d, df_t], axis=1)
    for i in extrap_columns:
        df[i] = df.apply(
            lambda r: extrapolate_energies(2, 3, r[i + " (DZ)"], r[i + " (TZ)"]), axis=1
        )

    df_subset = df[extrap_columns]
    print(df_subset)
    for i, j in zip(df_tmp.to_list(), df_subset[extrap_columns[0]].to_list()):
        print(i, j)


    # I need to store that as a series inside a dataframe df_dt ... if I
    # initialize it as empty first can I compute it directly to the desired
    # location?  Or else I need to copy the computed series into the dataframe
    return


if __name__ == "__main__":
    main()


