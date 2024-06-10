import pandas as pd
from glob import glob
from pprint import pprint as pp
import os
import extrapolator 

def og():
    base_path = "../sapt-components-2022/si/sapt_ref_data/"
    fs_adz = glob(f"{base_path}/adz/*.pkl")
    fs_atz = glob(f"{base_path}/atz/*.pkl")
    fs_qz = glob(f"{base_path}/aqz/*.pkl")
    extrapolator.extrapolator.generate_output_pkls(fs_adz, fs_atz, fs_qz)
    return

def main():
    base_path = "../../si/sapt_ref_data"
    fs_adz = []
    fs_atz = glob(f"{base_path}/atz_subset_358.pkl")
    fs_qz = glob(f"{base_path}/aqz_subset_358.pkl")
    print(fs_atz, fs_qz)
    extrapolator.extrapolator.generate_output_pkls(fs_adz, fs_atz, fs_qz, True)
    return


if __name__ == "__main__":
    main()
