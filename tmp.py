import pandas as pd
from glob import glob
from pprint import pprint as pp
import os
import extrapolator 

def main():
    base_path = "../sapt-components-2022/si/sapt_ref_data/"
    fs_adz = glob(f"{base_path}/adz/*.pkl")
    fs_atz = glob(f"{base_path}/atz/*.pkl")
    fs_qz = glob(f"{base_path}/aqz/*.pkl")
    extrapolator.extrapolator.generate_output_pkls(fs_adz, fs_atz, fs_qz)
    return


if __name__ == "__main__":
    main()
