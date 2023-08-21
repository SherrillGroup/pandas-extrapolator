import numpy as np
import pandas as pd
from glob import glob
import os
from pprint import pprint as pp
import src

extrap_columns_all = [
    "SAPT DISP20 ENERGY",
    "SAPT DISP21 ENERGY",
    "SAPT DISP22(S)(CCD) ENERGY",
    "SAPT DISP2(CCD) ENERGY",
    "SAPT DISP22(SDQ) ENERGY",
    "SAPT DISP22(T) ENERGY",
    "SAPT DISP22(T)(CCD) ENERGY",
    "SAPT DISP30 ENERGY",
    "SAPT ELST12,R ENERGY",
    "SAPT ELST13,R ENERGY",
    "SAPT EST.DISP22(T) ENERGY",
    "SAPT EST.DISP22(T)(CCD) ENERGY",
    "SAPT EXCH-DISP20 ENERGY",
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

extrap_columns = [
    # "SAPT DISP20 ENERGY",
    "SAPT DISP21 ENERGY",
    "SAPT DISP22(S)(CCD) ENERGY",
    # "SAPT DISP2(CCD) ENERGY",
    "SAPT DISP22(SDQ) ENERGY",
    "SAPT DISP22(T) ENERGY",
    "SAPT DISP22(T)(CCD) ENERGY",
    "SAPT DISP30 ENERGY",
    "SAPT ELST12,R ENERGY",
    "SAPT ELST13,R ENERGY",
    "SAPT EST.DISP22(T) ENERGY",
    # "SAPT EST.DISP22(T)(CCD) ENERGY",
    "SAPT EXCH-DISP20 ENERGY",
    "SAPT EXCH-DISP30 ENERGY",
    # "SAPT EXCH-IND-DISP30 ENERGY",
    "SAPT EXCH-IND22 ENERGY",
    "SAPT EXCH-IND30,R ENERGY",
    "SAPT EXCH11(S^2) ENERGY",
    "SAPT EXCH12(S^2) ENERGY",
    # "SAPT IND-DISP30 ENERGY",
    "SAPT IND22 ENERGY",
    "SAPT IND30,R ENERGY",
    "SAPT MP2 CORRELATION ENERGY",  # supermolecular MP2 E_corr for dMP2
]

def gather_dfs():
    atqz_pkls = [
        "sapt_ref_data/atqz/hbc6-plat-atqz-all.pkl",
        "sapt_ref_data/atqz/ion43-plat-atqz.pkl",
        "sapt_ref_data/atqz/s66x8-plat-atqz.pkl",
    ]
    adtz_pkls = [
        "sapt_ref_data/adtz/hbc6-plat-adtz-all.pkl",
        "sapt_ref_data/adtz/ion43-plat-adtz.pkl",
        "sapt_ref_data/adtz/s66x8-plat-adtz.pkl",
    ]
    frames = []
    for pkl in atqz_pkls:
        df = pd.read_pickle(pkl)
        frames.append(df)
    df_tq = pd.concat(frames)
    frames = []
    for pkl in adtz_pkls:
        df = pd.read_pickle(pkl)
        frames.append(df)
    df_dt = pd.concat(frames)
    df_dt.columns = df_dt.columns.values + f" (adtz)"
    df_tq.columns = df_tq.columns.values + f" (atqz)"
    df = pd.concat([df_dt, df_tq], axis=1)
    return df


def stats_split_by_extrap_col():
    stats = []
    df = gather_dfs()
    df = df.apply(lambda c: c * 627.509)
    for c in extrap_columns:
        sub_stats = []
        # sub_stats_c = []
        for i in np.arange(0.8, 1.2, 0.01):
            c_dt = c + " (adtz)"
            c_tq = c + " (atqz)"
            df["diff"] = (df[c_tq] - df[c_dt] * i)
            df["diff_og"] = (df[c_tq] - df[c_dt])
            mae = df["diff"].abs().mean()
            mae_og = df["diff_og"].abs().mean()
            rmse = np.sqrt((df["diff"] ** 2).mean())
            sub_stats.append([i, mae, rmse, mae_og])
            # sub_stats_c.append([i, mae, rmse, c])
        sub_stats.sort(key=lambda x: x[1])
        sub_stats = np.array(sub_stats)
        opt_param = sub_stats[0, 0]
        if abs(opt_param - 1.0) < 1e-14:
            print(c, 'opt_param == 1.0')
        print(c)
        pp(sub_stats[:3])
        mae_mae = np.array(sub_stats)[:, 1].mean()
        rmse_mae = np.array(sub_stats)[:, 2].mean()
        mae_mae_og = np.array(sub_stats)[:, 3].mean()
        stats.append([i, mae_mae, rmse_mae, mae_mae_og])
    stats.sort(key=lambda x: x[1])
    # pp(stats[:10])
    # pp(stats)
    opt_param = stats[0][0]
    print(opt_param)
    pd.set_option("display.max_rows", None)
    df2 = df.copy()
    df2.rename(columns={c: c.replace(" (adtz)", "") for c in df2.columns}, inplace=True)
    for c in extrap_columns:
        df2[c] = df2[c] * opt_param
    df2 = src.compute_sapt_terms.compute_sapt_terms(df2)
    levels_of_sapt = [
        'SAPT0 TOTAL ENERGY',
        'SSAPT0 TOTAL ENERGY',
        'SAPT2 TOTAL ENERGY',
        'SAPT2+ TOTAL ENERGY',
        'SAPT2+(CCD) TOTAL ENERGY',
        'SAPT2+DMP2 TOTAL ENERGY',
        'SAPT2+(CCD)DMP2 TOTAL ENERGY',
        'SAPT2+(3) TOTAL ENERGY',
        'SAPT2+(3)(CCD) TOTAL ENERGY',
        'SAPT2+(3)DMP2 TOTAL ENERGY',
        'SAPT2+(3)(CCD)DMP2 TOTAL ENERGY',
        'SAPT2+3 TOTAL ENERGY',
        'SAPT2+3(CCD) TOTAL ENERGY',
        'SAPT2+3DMP2 TOTAL ENERGY',
        'SAPT2+3(CCD)DMP2 TOTAL ENERGY',
    ]

    for c in levels_of_sapt:
        df[c + "_diff"] = df[c + " (atqz)"] - df[c + " (adtz)"]
        rmse1 = np.sqrt((df[c + "_diff"] ** 2).mean())
        df2[c + "_diff_scaled"] = df2[c + " (atqz)"] - df2[c]
        rmse2 = np.sqrt((df2[c + "_diff_scaled"] ** 2).mean())
        print(f"{c}:\n    RMSE unscaled: {rmse1:.6f}\n    RMSE scaled:   {rmse2:.6f}")
    return

def stats_split_by_extrap_col_opt_per_extrap_col():
    stats = []
    df = gather_dfs()
    df = df.apply(lambda c: c * 627.509)
    opt_params = {}
    for c in extrap_columns:
        sub_stats = []
        # sub_stats_c = []
        for i in np.arange(0.8, 1.2, 0.01):
            c_dt = c + " (adtz)"
            c_tq = c + " (atqz)"
            df["diff"] = (df[c_tq] - df[c_dt] * i)
            df["diff_og"] = (df[c_tq] - df[c_dt])
            mae = df["diff"].abs().mean()
            mae_og = df["diff_og"].abs().mean()
            rmse = np.sqrt((df["diff"] ** 2).mean())
            sub_stats.append([i, mae, rmse, mae_og])
            # sub_stats_c.append([i, mae, rmse, c])
        sub_stats.sort(key=lambda x: x[1])
        sub_stats = np.array(sub_stats)
        opt_param = sub_stats[0, 0]
        opt_params[c] = opt_param
        if abs(opt_param - 1.0) < 1e-14:
            print(c, 'opt_param == 1.0')
        print(c)
        pp(sub_stats[:3])
        mae_mae = np.array(sub_stats)[:, 1].mean()
        rmse_mae = np.array(sub_stats)[:, 2].mean()
        mae_mae_og = np.array(sub_stats)[:, 3].mean()
        stats.append([i, mae_mae, rmse_mae, mae_mae_og])
    stats.sort(key=lambda x: x[1])
    # pp(stats[:10])
    # pp(stats)
    opt_param = stats[0][0]
    print(opt_param)
    pd.set_option("display.max_rows", None)
    for i in df.columns.values:
        if i.endswith(" (adtz)"):
            i2 = i.replace(" (adtz)", "")
            df[i2] = df.apply(lambda r: r[i], axis=1)
    for c in extrap_columns:
        print("scaling", c, "by", opt_params[c])
        df[c] = df[c] * opt_params[c]
    df = src.compute_sapt_terms.compute_sapt_terms(df)
    # print(len(df.columns.values), df.columns.values)
    # print(len(df.columns.values), df.columns.values)
    levels_of_sapt = [
        'SAPT0 TOTAL ENERGY',
        'SSAPT0 TOTAL ENERGY',
        'SAPT2 TOTAL ENERGY',
        'SAPT2+ TOTAL ENERGY',
        'SAPT2+(CCD) TOTAL ENERGY',
        'SAPT2+DMP2 TOTAL ENERGY',
        'SAPT2+(CCD)DMP2 TOTAL ENERGY',
        'SAPT2+(3) TOTAL ENERGY',
        'SAPT2+(3)(CCD) TOTAL ENERGY',
        'SAPT2+(3)DMP2 TOTAL ENERGY',
        'SAPT2+(3)(CCD)DMP2 TOTAL ENERGY',
        'SAPT2+3 TOTAL ENERGY',
        'SAPT2+3(CCD) TOTAL ENERGY',
        'SAPT2+3DMP2 TOTAL ENERGY',
        'SAPT2+3(CCD)DMP2 TOTAL ENERGY',
    ]

    for c in levels_of_sapt:
        df[c + "_diff"] = df[c + " (atqz)"] - df[c + " (adtz)"]
        rmse1 = np.sqrt((df[c + "_diff"] ** 2).mean())
        df[c + "_diff_scaled"] = df[c + " (atqz)"] - df[c]
        rmse2 = np.sqrt((df[c + "_diff_scaled"] ** 2).mean())
        print(f"{c}:\n    RMSE unscaled: {rmse1:.6f}\n    RMSE scaled:   {rmse2:.6f}")
        # r1 = df2.iloc[0]
        # tqz = r1[f'{c} (atqz)']
        # dtz = r1[f'{c}']
        # dtz_scaled = dtz * opt_param
        # print(f"    TQZ: {tqz:.6f}\n    DTZ: {dtz:.6f}\n    DTZ Scaled: {dtz_scaled:.6f}")



    return

def stats_scale_by_total_energy():
    stats = []
    df = gather_dfs()
    df = df.apply(lambda c: c * 627.509)
    opt_params = {}
    levels_of_sapt = [
        'SAPT0 TOTAL ENERGY',
        'SSAPT0 TOTAL ENERGY',
        'SAPT2 TOTAL ENERGY',
        'SAPT2+ TOTAL ENERGY',
        'SAPT2+(CCD) TOTAL ENERGY',
        'SAPT2+DMP2 TOTAL ENERGY',
        'SAPT2+(CCD)DMP2 TOTAL ENERGY',
        'SAPT2+(3) TOTAL ENERGY',
        'SAPT2+(3)(CCD) TOTAL ENERGY',
        'SAPT2+(3)DMP2 TOTAL ENERGY',
        'SAPT2+(3)(CCD)DMP2 TOTAL ENERGY',
        'SAPT2+3 TOTAL ENERGY',
        'SAPT2+3(CCD) TOTAL ENERGY',
        'SAPT2+3DMP2 TOTAL ENERGY',
        'SAPT2+3(CCD)DMP2 TOTAL ENERGY',
    ]
    print(df.columns.values)
    for c in levels_of_sapt:
        c_dt = c + " (adtz)"
        c_tq = c + " (atqz)"
        df[c + " unscaled"] = df.apply(lambda r: r[c_dt], axis=1)
        sub_stats = []
        # sub_stats_c = []
        for i in np.arange(0.98, 1.07, 0.0001):
            df["diff"] = (df[c_tq] - df[c_dt] * i)
            df["diff_og"] = (df[c_tq] - df[c_dt])
            mae = df["diff"].abs().mean()
            mae_og = df["diff_og"].abs().mean()
            rmse = np.sqrt((df["diff"] ** 2).mean())
            sub_stats.append([i, mae, rmse, mae_og])
            # sub_stats_c.append([i, mae, rmse, c])
        sub_stats.sort(key=lambda x: x[1])
        sub_stats = np.array(sub_stats)
        opt_param = sub_stats[0, 0]
        opt_params[c] = opt_param
        if abs(opt_param - 1.0) < 1e-14:
            print(c, 'opt_param == 1.0')
        else:
            print(c)
            pp(sub_stats[:3])
        mae_mae = np.array(sub_stats)[:, 1].mean()
        rmse_mae = np.array(sub_stats)[:, 2].mean()
        mae_mae_og = np.array(sub_stats)[:, 3].mean()
        stats.append([i, mae_mae, rmse_mae, mae_mae_og])
    stats.sort(key=lambda x: x[1])
    opt_param = stats[0][0]
    print(opt_param)
    pd.set_option("display.max_rows", None)
    for i in df.columns.values:
        if i.endswith(" (adtz)"):
            i2 = i.replace(" (adtz)", "")
            df[i2] = df.apply(lambda r: r[i], axis=1)
    for c in levels_of_sapt:
        print("scaling", c, "by", opt_params[c])
        df[c] = df[c] * opt_params[c]

    for c in levels_of_sapt:
        df[c + "_diff"] = df[c + " (atqz)"] - df[c + " (adtz)"]
        df[c + "_diff_scaled"] = df[c + " (atqz)"] - df[c]
        rmse1 = np.sqrt((df[c + "_diff"] ** 2).mean())
        rmse2 = np.sqrt((df[c + "_diff_scaled"] ** 2).mean())
        mae1 = df[c + "_diff"].abs().mean()
        mae2 = df[c + "_diff_scaled"].abs().mean()
        print(f"{c}:\n    RMSE unscaled: {rmse1:.6f}\n    RMSE scaled:   {rmse2:.6f}")
        print(f"    MAE unscaled: {mae1:.6f}\n    MAE scaled:   {mae2:.6f}")
    return

def extrap_plotting():
    levels_of_sapt = [
        'SAPT0 TOTAL ENERGY',
        'SSAPT0 TOTAL ENERGY',
        'SAPT2 TOTAL ENERGY',
        'SAPT2+ TOTAL ENERGY',
        'SAPT2+(CCD) TOTAL ENERGY',
        'SAPT2+DMP2 TOTAL ENERGY',
        'SAPT2+(CCD)DMP2 TOTAL ENERGY',
        'SAPT2+(3) TOTAL ENERGY',
        'SAPT2+(3)(CCD) TOTAL ENERGY',
        'SAPT2+(3)DMP2 TOTAL ENERGY',
        'SAPT2+(3)(CCD)DMP2 TOTAL ENERGY',
        'SAPT2+3 TOTAL ENERGY',
        'SAPT2+3(CCD) TOTAL ENERGY',
        'SAPT2+3DMP2 TOTAL ENERGY',
        'SAPT2+3(CCD)DMP2 TOTAL ENERGY',
    ]
    # TODO: only look at qz data ones...
    basis_set = [
            "dz",
            "tz",
            "qz",
            "dt",
            "tq",
            ]
    # TODO: compare with reference values CCSD...
    # NOTE: standard extrapolation equations might not be the best for SAPT
    return


def main():
    # NOTE: Energy Units are Hartrees
    # stats_split_by_extrap_col()
    stats_split_by_extrap_col_opt_per_extrap_col()
    # stats_scale_by_total_energy()
    return


if __name__ == "__main__":
    main()
