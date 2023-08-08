import numpy as np
import pandas as pd
from glob import glob
import os
from pprint import pprint as pp


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


def og_stats():
    stats = []
    for i in np.arange(0.7, 1.5, 0.01):
        sub_stats = []
        for c in extrap_columns:
            c_dt = c + " (adtz)"
            c_tq = c + " (atqz)"
            df['diff'] = df[c_tq] - df[c_dt] * i
            mae = df['diff'].abs().mean()
            rmse = np.sqrt((df['diff']**2).mean())
            sub_stats.append([i, mae, rmse])
        mae_mae = np.array(sub_stats)[:, 1].mean()
        rmse_mae = np.array(sub_stats)[:, 2].mean()
        stats.append([i, mae_mae, rmse_mae])
    pp(stats)
    return


def main():
    extrap_columns = [
        "SAPT DISP20 ENERGY",
        "SAPT DISP21 ENERGY",
        "SAPT DISP22(S)(CCD) ENERGY",
        'SAPT DISP2(CCD) ENERGY',
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
    df = gather_dfs()
    df['extrap_sum_adtz'] = [0 for i in range(len(df))]
    df['extrap_sum_atqz'] = [0 for i in range(len(df))]
    for c in extrap_columns:
        df['extrap_sum_adtz'] += df[c + " (adtz)"]
        df['extrap_sum_atqz'] += df[c + " (atqz)"]
    stats = []
    for i in np.arange(0.95, 1.05, 0.001):
        df['diff'] = df['extrap_sum_atqz'] - df['extrap_sum_adtz'] * i
        mae = df['diff'].abs().mean()
        rmse = np.sqrt((df['diff']**2).mean())
        stats.append([i, mae, rmse])
    df['diff'] = df['extrap_sum_atqz'] - df['extrap_sum_adtz']
    mae = df['diff'].abs().mean()
    rmse = np.sqrt((df['diff']**2).mean())
    print(f"Unscaled MAE: {mae}")
    print(f"Unscaled RMSE: {rmse}")
    print("sorted by MAE")
    stats.sort(key=lambda x: x[1])
    pp(stats)
    print(f"Unscaled MAE: {mae}")
    print(f"Unscaled RMSE: {rmse}")
    print("sorted by RMSE")
    stats.sort(key=lambda x: x[2])
    pp(stats)
    return


if __name__ == "__main__":
    main()
