#!/usr/bin/env python3
"""Python translation of Democracy-AER-v2.R.

This script reproduces the main workflow from the original R replication code:
1) Descriptive statistics
2) Two-way fixed effects estimation with clustered SEs
3) Split-panel jackknife bias correction
4) Difference-GMM (Arellano-Bond style) using one-step IV-GMM
5) Anderson-Hsiao IV estimation with split-sample bias corrections
6) Panel bootstrap robust standard errors
7) Final summary table

The script assumes the data file democracy-balanced-l4.dta is in the same folder
as this file.
"""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import IVGMM
from linearmodels.panel import PanelOLS
from scipy.linalg import qr
from scipy.stats import norm


FE_PARAM_NAMES = ["dem", "lgdp_l1", "lgdp_l2", "lgdp_l3", "lgdp_l4"]
AB_PARAM_NAMES = ["d_dem", "d_lgdp_l1", "d_lgdp_l2", "d_lgdp_l3", "d_lgdp_l4"]
ROW_NAMES = [
    "Democracy",
    "CSE",
    "BSE",
    "L1.log(gdp)",
    "CSE1",
    "BSE1",
    "L2.log(gdp)",
    "CSE2",
    "BSE2",
    "L3.log(gdp)",
    "CSE3",
    "BSE3",
    "L4.log(gdp)",
    "CSE4",
    "BSE4",
    "LR-Democracy",
    "CSE5",
    "BSE5",
]
COL_NAMES = [
    "FE",
    "SBC",
    "AB",
    "AB-SBC1",
    "AB-SBC5",
    "AH",
    "AH-SBC1",
    "AH-SBC5",
]


@dataclass
class EstimationResult:
    coefs: np.ndarray
    cov: np.ndarray
    cse: np.ndarray
    lr: float
    cse_lr: float
    n_moments: int | None = None
    n_exog: int | None = None
    n_excluded_instr: int | None = None


def robust_sd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    q75 = np.quantile(x, 0.75)
    q25 = np.quantile(x, 0.25)
    return float((q75 - q25) / (norm.ppf(0.75) - norm.ppf(0.25)))


def compute_lr_and_se(coefs: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    denom = 1.0 - float(np.sum(coefs[1:5]))
    lr = float(coefs[0] / denom)
    jac = np.array([1.0, lr, lr, lr, lr], dtype=float) / denom
    var_lr = float(jac @ cov @ jac)
    cse_lr = float(np.sqrt(max(var_lr, 0.0)))
    return lr, cse_lr


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_stata(path)
    data["id"] = data["id"].astype(int)
    data["year"] = data["year"].astype(int)
    data = data.sort_values(["id", "year"]).reset_index(drop=True)
    return data


def descriptive_stats(data: pd.DataFrame) -> pd.DataFrame:
    cols = ["dem", "lgdp"]
    stats = np.column_stack(
        [
            data[cols].mean().to_numpy(),
            data[cols].std(ddof=1).to_numpy(),
            data.loc[data["dem"] == 1, cols].mean().to_numpy(),
            data.loc[data["dem"] == 0, cols].mean().to_numpy(),
        ]
    )
    counts = np.array(
        [
            float(len(data)),
            float(len(data)),
            float((data["dem"] == 1).sum()),
            float((data["dem"] == 0).sum()),
        ]
    )
    out = np.vstack([stats, counts])
    return pd.DataFrame(
        out,
        index=["Democracy", "Log(GDP)", "Number Obs."],
        columns=["Mean", "SD", "Dem = 1", "Dem = 0"],
    )


def add_lags(df: pd.DataFrame, var: str, max_lag: int) -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby("id", sort=False)[var]
    for lag in range(1, max_lag + 1):
        out[f"{var}_l{lag}"] = grp.shift(lag)
    return out


def filter_instruments_full_rank(z: pd.DataFrame, exog: pd.DataFrame) -> pd.DataFrame:
    if z.shape[1] == 0:
        return z

    ex = np.asarray(exog, dtype=float)
    zz = np.asarray(z, dtype=float)
    a = np.column_stack([ex, zz])
    qr_out = qr(a, mode="economic", pivoting=True)
    r = qr_out[1] if len(qr_out) > 1 else np.empty((0, 0))
    piv = qr_out[2] if len(qr_out) > 2 else np.arange(a.shape[1], dtype=int)
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        return z.iloc[:, []]

    tol = np.max(a.shape) * np.finfo(float).eps * diag[0]
    rank = int(np.sum(diag > tol))
    n_exog = ex.shape[1]

    keep = [j - n_exog for j in piv[:rank] if j >= n_exog]
    keep = sorted(set(keep))
    if not keep:
        return z.iloc[:, []]
    return z.iloc[:, keep]


def split_masks_by_year_rank(
    df: pd.DataFrame, rank_cutoff: int = 14
) -> tuple[pd.Series, pd.Series]:
    years = np.sort(df["year"].unique())
    year_rank = {year: i + 1 for i, year in enumerate(years)}
    rank_series = df["year"].map(year_rank)
    cutoff = (
        rank_cutoff if len(years) >= rank_cutoff else int(math.ceil(len(years) / 2))
    )
    mask1 = rank_series <= cutoff
    mask2 = rank_series >= cutoff
    return mask1, mask2


def fit_fe(data: pd.DataFrame) -> EstimationResult:
    df = add_lags(data, "lgdp", 4)
    df = df.set_index(["id", "year"])
    keep_cols = ["lgdp", *FE_PARAM_NAMES]
    reg = df.dropna(subset=keep_cols)

    model = PanelOLS(
        dependent=reg["lgdp"],
        exog=reg[FE_PARAM_NAMES],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=False,
    )
    fit = model.fit(cov_type="clustered", cluster_entity=True)

    coefs = fit.params.loc[FE_PARAM_NAMES].to_numpy(dtype=float)
    cov = fit.cov.loc[FE_PARAM_NAMES, FE_PARAM_NAMES].to_numpy(dtype=float)
    cse = np.sqrt(np.diag(cov))
    lr, cse_lr = compute_lr_and_se(coefs, cov)

    return EstimationResult(coefs=coefs, cov=cov, cse=cse, lr=lr, cse_lr=cse_lr)


def fit_ab(data: pd.DataFrame, max_instr_lag: int) -> EstimationResult:
    df = add_lags(data, "lgdp", max(max_instr_lag, 5))
    df = add_lags(df, "dem", max_instr_lag)

    grp = df.groupby("id", sort=False)
    df["d_lgdp"] = grp["lgdp"].diff()
    df["d_dem"] = grp["dem"].diff()
    for k in range(1, 5):
        df[f"d_lgdp_l{k}"] = df[f"lgdp_l{k}"] - df[f"lgdp_l{k+1}"]

    req = ["d_lgdp", *AB_PARAM_NAMES]
    reg = df.dropna(subset=req).copy()
    reg = reg.sort_values(["id", "year"]).reset_index(drop=True)
    reg["teff"] = reg.groupby("id", sort=False).cumcount() + 1

    # Ensure all lags needed for X_{t-h} = (D_{t-h}, Y_{t-h-1}, ..., Y_{t-h-4}) exist.
    t_eff = int(reg["teff"].max())
    max_h_needed = min(t_eff - 1, max_instr_lag)
    max_y_lag_needed = max_h_needed + 4
    for lag in range(1, max_h_needed + 1):
        col = f"dem_l{lag}"
        if col not in reg.columns:
            reg[col] = reg.groupby("id", sort=False)["dem"].shift(lag)
    for lag in range(1, max_y_lag_needed + 1):
        col = f"lgdp_l{lag}"
        if col not in reg.columns:
            reg[col] = reg.groupby("id", sort=False)["lgdp"].shift(lag)

    y = reg["d_lgdp"]
    endog = reg[AB_PARAM_NAMES]
    exog = pd.DataFrame(index=reg.index)

    # Equation (8.19): g_i(Z_i,γ) = { (ΔY_it - ΔX'_it γ) X_i^{t-1} }.
    # With fixed 5-dimensional X_t = (D_t, Y_{t-1}, ..., Y_{t-4}),
    # X_i^{t-1} stacks X_{t-1}, X_{t-2}, ..., X_1.
    z_dict: dict[str, np.ndarray] = {}
    teff = reg["teff"].to_numpy()
    for t in range(2, t_eff + 1):
        mask_t = teff == t
        h_max = min(t - 1, max_instr_lag)
        for h in range(1, h_max + 1):
            x_vals = {
                f"D_h{h}": reg[f"dem_l{h}"].to_numpy(),
                f"Y1_h{h}": reg[f"lgdp_l{h+1}"].to_numpy(),
                f"Y2_h{h}": reg[f"lgdp_l{h+2}"].to_numpy(),
                f"Y3_h{h}": reg[f"lgdp_l{h+3}"].to_numpy(),
                f"Y4_h{h}": reg[f"lgdp_l{h+4}"].to_numpy(),
            }
            for base_name, vals in x_vals.items():
                col = np.where(mask_t, vals, 0.0)
                col = np.nan_to_num(col, nan=0.0)
                if np.var(col) > 0:
                    z_dict[f"z_t{t}_{base_name}"] = col

    z = pd.DataFrame(z_dict, index=reg.index)

    # Remove all-zero and duplicate columns for numerical stability.
    z = z.loc[:, z.var(axis=0) > 0]
    z = z.T.drop_duplicates().T
    z = filter_instruments_full_rank(z, exog)

    model = IVGMM(y, exog=None, endog=endog, instruments=z)
    fit = model.fit(
        iter_limit=1,
        cov_type="clustered",
        clusters=reg["id"].to_numpy(),
    )

    coefs = fit.params.loc[AB_PARAM_NAMES].to_numpy(dtype=float)
    cov = fit.cov.loc[AB_PARAM_NAMES, AB_PARAM_NAMES].to_numpy(dtype=float)
    cse = np.sqrt(np.diag(cov))
    lr, cse_lr = compute_lr_and_se(coefs, cov)
    n_exog = 0
    n_excluded_instr = int(z.shape[1])
    n_moments = n_excluded_instr

    return EstimationResult(
        coefs=coefs,
        cov=cov,
        cse=cse,
        lr=lr,
        cse_lr=cse_lr,
        n_moments=n_moments,
        n_exog=n_exog,
        n_excluded_instr=n_excluded_instr,
    )


def fit_ah(data: pd.DataFrame, max_instr_lag: int) -> EstimationResult:
    # AH score moments use lagged structural regressors only:
    # E[(Δy_it - ΔX'_it γ) X_{i,t-1}] = 0,
    # with X_{i,t-1} = (D_{i,t-1}, Y_{i,t-2}, ..., Y_{i,t-5}).
    df = add_lags(data, "lgdp", max(5, max_instr_lag))
    df = add_lags(df, "dem", max(1, max_instr_lag))

    grp = df.groupby("id", sort=False)
    df["d_lgdp"] = grp["lgdp"].diff()
    df["d_dem"] = grp["dem"].diff()
    for k in range(1, 5):
        df[f"d_lgdp_l{k}"] = df[f"lgdp_l{k}"] - df[f"lgdp_l{k+1}"]

    req = [
        "d_lgdp",
        *AB_PARAM_NAMES,
        "dem_l1",
        "lgdp_l2",
        "lgdp_l3",
        "lgdp_l4",
        "lgdp_l5",
    ]
    reg = df.dropna(subset=req).copy()
    reg = reg.sort_values(["id", "year"]).reset_index(drop=True)
    reg["teff"] = reg.groupby("id", sort=False).cumcount() + 1

    y = reg["d_lgdp"]
    endog = reg[AB_PARAM_NAMES]

    exog = pd.DataFrame(index=reg.index)

    # Equation (8.20): ĝ_i(Z_i,γ) = { (ΔY_it - ΔX'_it γ) X_{i(t-1)} }.
    # With fixed 5-dimensional X_{t-1} = (D_{t-1}, Y_{t-2}, ..., Y_{t-5}),
    # uncollapsed score moments are period-specific interactions.
    t_eff = int(reg["teff"].max())
    teff = reg["teff"].to_numpy()
    x_lag = {
        "D_l1": reg["dem_l1"].to_numpy(),
        "Y_l2": reg["lgdp_l2"].to_numpy(),
        "Y_l3": reg["lgdp_l3"].to_numpy(),
        "Y_l4": reg["lgdp_l4"].to_numpy(),
        "Y_l5": reg["lgdp_l5"].to_numpy(),
    }
    z_dict: dict[str, np.ndarray] = {}
    for t in range(2, t_eff + 1):
        mask_t = teff == t
        for base_name, vals in x_lag.items():
            col = np.where(mask_t, vals, 0.0)
            col = np.nan_to_num(col, nan=0.0)
            if np.var(col) > 0:
                z_dict[f"z_t{t}_{base_name}"] = col

    z = pd.DataFrame(z_dict, index=reg.index)
    z = z.loc[:, z.var(axis=0) > 0]
    z = z.T.drop_duplicates().T
    z = filter_instruments_full_rank(z, exog)

    model = IVGMM(y, exog=None, endog=endog, instruments=z)
    fit = model.fit(
        iter_limit=1,
        cov_type="clustered",
        clusters=reg["id"].to_numpy(),
    )

    coefs = fit.params.loc[AB_PARAM_NAMES].to_numpy(dtype=float)
    cov = fit.cov.loc[AB_PARAM_NAMES, AB_PARAM_NAMES].to_numpy(dtype=float)
    cse = np.sqrt(np.diag(cov))
    lr, cse_lr = compute_lr_and_se(coefs, cov)
    n_exog = 0
    n_excluded_instr = int(z.shape[1])
    n_moments = n_excluded_instr

    return EstimationResult(
        coefs=coefs,
        cov=cov,
        cse=cse,
        lr=lr,
        cse_lr=cse_lr,
        n_moments=n_moments,
        n_exog=n_exog,
        n_excluded_instr=n_excluded_instr,
    )


def split_sample_ab_correction(
    data: pd.DataFrame,
    base_ab: EstimationResult,
    partitions: int,
    max_instr_lag: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    ids = np.array(sorted(data["id"].unique()))
    n = len(ids)

    avg_coefs = np.zeros_like(base_ab.coefs)
    avg_lr = 0.0

    for _ in range(partitions):
        sample1 = set(
            rng.choice(ids, size=int(math.ceil(n / 2.0)), replace=False).tolist()
        )
        d1 = data[data["id"].isin(sample1)].copy()
        d2 = data[~data["id"].isin(sample1)].copy()

        ab1 = fit_ab(d1, max_instr_lag=max_instr_lag)
        ab2 = fit_ab(d2, max_instr_lag=max_instr_lag)

        avg_coefs += 0.5 * (ab1.coefs + ab2.coefs) / partitions
        avg_lr += 0.5 * (ab1.lr + ab2.lr) / partitions

    coefs_jbc = 2.0 * base_ab.coefs - avg_coefs
    lr_jbc = 2.0 * base_ab.lr - avg_lr
    return coefs_jbc, lr_jbc


def split_sample_ah_correction(
    data: pd.DataFrame,
    base_ah: EstimationResult,
    partitions: int,
    max_instr_lag: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    ids = np.array(sorted(data["id"].unique()))
    n = len(ids)

    avg_coefs = np.zeros_like(base_ah.coefs)
    avg_lr = 0.0

    for _ in range(partitions):
        sample1 = set(
            rng.choice(ids, size=int(math.ceil(n / 2.0)), replace=False).tolist()
        )
        d1 = data[data["id"].isin(sample1)].copy()
        d2 = data[~data["id"].isin(sample1)].copy()

        ah1 = fit_ah(d1, max_instr_lag=max_instr_lag)
        ah2 = fit_ah(d2, max_instr_lag=max_instr_lag)

        avg_coefs += 0.5 * (ah1.coefs + ah2.coefs) / partitions
        avg_lr += 0.5 * (ah1.lr + ah2.lr) / partitions

    coefs_jbc = 2.0 * base_ah.coefs - avg_coefs
    lr_jbc = 2.0 * base_ah.lr - avg_lr
    return coefs_jbc, lr_jbc


def bootstrap_resample(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    ids = np.array(sorted(data["id"].unique()))
    years = np.array(sorted(data["year"].unique()))

    sampled_ids = rng.choice(ids, size=len(ids), replace=True)
    out_parts: list[pd.DataFrame] = []

    for new_id, orig_id in enumerate(sampled_ids, start=1):
        panel = data[data["id"] == orig_id].sort_values("year").copy()
        panel["id"] = new_id
        panel["year"] = years
        out_parts.append(panel)

    out = pd.concat(out_parts, axis=0, ignore_index=True)
    out = out.sort_values(["id", "year"]).reset_index(drop=True)
    return out


def boot_stat_fe(data: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    try:
        fe = fit_fe(data)
        mask1, mask2 = split_masks_by_year_rank(data, rank_cutoff=14)
        fe1 = fit_fe(data[mask1].copy())
        fe2 = fit_fe(data[mask2].copy())

        coefs_jbc = 19.0 * fe.coefs / 9.0 - 10.0 * (fe1.coefs + fe2.coefs) / 18.0
        lr_jbc = 19.0 * fe.lr / 9.0 - 10.0 * (fe1.lr + fe2.lr) / 18.0

        return np.concatenate(
            [
                fe.coefs,
                coefs_jbc,
                np.array([fe.lr, lr_jbc], dtype=float),
            ]
        )
    except Exception:
        return np.full(12, np.nan, dtype=float)


def boot_stat_ab(
    data: pd.DataFrame, rng: np.random.Generator, max_instr_lag: int
) -> np.ndarray:
    try:
        ab = fit_ab(data, max_instr_lag=max_instr_lag)
        coefs_jbc, lr_jbc = split_sample_ab_correction(
            data,
            base_ab=ab,
            partitions=1,
            max_instr_lag=max_instr_lag,
            rng=rng,
        )
        coefs_jbc5, lr_jbc5 = split_sample_ab_correction(
            data,
            base_ab=ab,
            partitions=5,
            max_instr_lag=max_instr_lag,
            rng=rng,
        )
        return np.concatenate(
            [
                ab.coefs,
                coefs_jbc,
                coefs_jbc5,
                np.array([ab.lr, lr_jbc, lr_jbc5], dtype=float),
            ]
        )
    except Exception:
        return np.full(18, np.nan, dtype=float)


def boot_stat_ah(
    data: pd.DataFrame, rng: np.random.Generator, max_instr_lag: int
) -> np.ndarray:
    try:
        ah = fit_ah(data, max_instr_lag=max_instr_lag)
        coefs_jbc, lr_jbc = split_sample_ah_correction(
            data,
            base_ah=ah,
            partitions=1,
            max_instr_lag=max_instr_lag,
            rng=rng,
        )
        coefs_jbc5, lr_jbc5 = split_sample_ah_correction(
            data,
            base_ah=ah,
            partitions=5,
            max_instr_lag=max_instr_lag,
            rng=rng,
        )
        return np.concatenate(
            [
                ah.coefs,
                coefs_jbc,
                coefs_jbc5,
                np.array([ah.lr, lr_jbc, lr_jbc5], dtype=float),
            ]
        )
    except Exception:
        return np.full(18, np.nan, dtype=float)


def run_bootstrap(
    data: pd.DataFrame,
    stat_name: str,
    reps: int,
    seed: int,
    max_instr_lag: int,
) -> np.ndarray:
    master = np.random.default_rng(seed)
    n_stats = 12 if stat_name == "fe" else 18
    out = np.full((reps, n_stats), np.nan, dtype=float)

    for r in range(reps):
        rng = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
        sample = bootstrap_resample(data, rng)
        if stat_name == "fe":
            out[r, :] = boot_stat_fe(sample, rng)
        elif stat_name == "ab":
            out[r, :] = boot_stat_ab(sample, rng, max_instr_lag=max_instr_lag)
        elif stat_name == "ah":
            out[r, :] = boot_stat_ah(sample, rng, max_instr_lag=max_instr_lag)
        else:
            raise ValueError(f"Unknown bootstrap statistic: {stat_name}")

    return out


def build_table(
    fe: EstimationResult,
    coefs_jbc_fe: np.ndarray,
    lr_jbc_fe: float,
    bse_fe: np.ndarray,
    bse_jbc_fe: np.ndarray,
    bse_lr_fe: float,
    bse_lr_jbc: float,
    ab: EstimationResult,
    coefs_ab_jbc: np.ndarray,
    coefs_ab_jbc5: np.ndarray,
    lr_ab_jbc: float,
    lr_ab_jbc5: float,
    bse_ab: np.ndarray,
    bse_ab_jbc: np.ndarray,
    bse_ab_jbc5: np.ndarray,
    bse_lr_ab: float,
    bse_lr_ab_jbc: float,
    bse_lr_ab_jbc5: float,
    ah: EstimationResult,
    coefs_ah_jbc: np.ndarray,
    coefs_ah_jbc5: np.ndarray,
    lr_ah_jbc: float,
    lr_ah_jbc5: float,
    bse_ah: np.ndarray,
    bse_ah_jbc: np.ndarray,
    bse_ah_jbc5: np.ndarray,
    bse_lr_ah: float,
    bse_lr_ah_jbc: float,
    bse_lr_ah_jbc5: float,
) -> pd.DataFrame:
    table = np.full((18, 8), np.nan, dtype=float)

    coef_rows = [0, 3, 6, 9, 12]
    cse_rows = [1, 4, 7, 10, 13]
    bse_rows = [2, 5, 8, 11, 14]

    table[coef_rows, 0] = fe.coefs
    table[cse_rows, 0] = fe.cse
    table[bse_rows, 0] = bse_fe

    table[coef_rows, 1] = coefs_jbc_fe
    table[bse_rows, 1] = bse_jbc_fe

    table[coef_rows, 2] = ab.coefs
    table[cse_rows, 2] = ab.cse
    table[bse_rows, 2] = bse_ab

    table[coef_rows, 3] = coefs_ab_jbc
    table[bse_rows, 3] = bse_ab_jbc

    table[coef_rows, 4] = coefs_ab_jbc5
    table[bse_rows, 4] = bse_ab_jbc5

    table[15, 0] = fe.lr
    table[16, 0] = fe.cse_lr
    table[17, 0] = bse_lr_fe

    table[15, 1] = lr_jbc_fe
    table[17, 1] = bse_lr_jbc

    table[15, 2] = ab.lr
    table[17, 2] = bse_lr_ab

    table[15, 3] = lr_ab_jbc
    table[16, 3] = ab.cse_lr
    table[17, 3] = bse_lr_ab_jbc

    table[15, 4] = lr_ab_jbc5
    table[17, 4] = bse_lr_ab_jbc5

    table[coef_rows, 5] = ah.coefs
    table[cse_rows, 5] = ah.cse
    table[bse_rows, 5] = bse_ah

    table[coef_rows, 6] = coefs_ah_jbc
    table[bse_rows, 6] = bse_ah_jbc

    table[coef_rows, 7] = coefs_ah_jbc5
    table[bse_rows, 7] = bse_ah_jbc5

    table[15, 5] = ah.lr
    table[16, 5] = ah.cse_lr
    table[17, 5] = bse_lr_ah

    table[15, 6] = lr_ah_jbc
    table[17, 6] = bse_lr_ah_jbc

    table[15, 7] = lr_ah_jbc5
    table[17, 7] = bse_lr_ah_jbc5

    table[[0, 1, 2, 15, 16, 17], :] *= 100.0

    return pd.DataFrame(table, index=ROW_NAMES, columns=COL_NAMES)


def write_latex_table(
    table: pd.DataFrame, output_path: Path, caption: str, label: str
) -> None:
    paper_table = table.copy()
    paper_table = paper_table.apply(
        lambda col: col.map(lambda x: "" if pd.isna(x) else f"{x:0.2f}")
    )
    latex = paper_table.to_latex(
        index=True,
        escape=False,
        na_rep="",
        column_format="l" + "r" * paper_table.shape[1],
        caption=caption,
        label=label,
        position="htbp",
        multicolumn=True,
        multicolumn_format="c",
    )
    latex += (
        "\n% Notes: Coefficients and long-run effects are in percentage points; "
        "CSE and BSE denote clustered and bootstrap standard errors, respectively.\n"
    )
    output_path.write_text(latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replication of Democracy-AER-v2.R in Python"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "democracy-balanced-l4.dta",
        help="Path to democracy-balanced-l4.dta",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=500,
        help="Number of panel bootstrap replications (R in the original script).",
    )
    parser.add_argument(
        "--max-instr-lag",
        type=int,
        default=22,
        help="Maximum lag depth for GMM instruments (uses lag 2+ for lgdp and lag 1+ for dem).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=888,
        help="Random seed.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip bootstrap and only run point estimates.",
    )
    parser.add_argument(
        "--latex-caption",
        type=str,
        default="Replication results: Causal impact of democracy on growth",
        help="Caption used in the LaTeX output table.",
    )
    parser.add_argument(
        "--latex-label",
        type=str,
        default="tab:democracy-replication-results",
        help="Label used in the LaTeX output table.",
    )
    args = parser.parse_args()

    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = load_data(data_path)

    dstat = descriptive_stats(data)

    fe = fit_fe(data)
    mask1, mask2 = split_masks_by_year_rank(data, rank_cutoff=14)
    fe1 = fit_fe(data[mask1].copy())
    fe2 = fit_fe(data[mask2].copy())

    coefs_jbc_fe = 19.0 * fe.coefs / 9.0 - 10.0 * (fe1.coefs + fe2.coefs) / 18.0
    lr_jbc_fe = 19.0 * fe.lr / 9.0 - 10.0 * (fe1.lr + fe2.lr) / 18.0

    rng_main = np.random.default_rng(args.seed)
    ab = fit_ab(data, max_instr_lag=args.max_instr_lag)
    coefs_ab_jbc, lr_ab_jbc = split_sample_ab_correction(
        data,
        base_ab=ab,
        partitions=1,
        max_instr_lag=args.max_instr_lag,
        rng=rng_main,
    )
    coefs_ab_jbc5, lr_ab_jbc5 = split_sample_ab_correction(
        data,
        base_ab=ab,
        partitions=5,
        max_instr_lag=args.max_instr_lag,
        rng=rng_main,
    )

    ah = fit_ah(data, max_instr_lag=args.max_instr_lag)
    coefs_ah_jbc, lr_ah_jbc = split_sample_ah_correction(
        data,
        base_ah=ah,
        partitions=1,
        max_instr_lag=args.max_instr_lag,
        rng=rng_main,
    )
    coefs_ah_jbc5, lr_ah_jbc5 = split_sample_ah_correction(
        data,
        base_ah=ah,
        partitions=5,
        max_instr_lag=args.max_instr_lag,
        rng=rng_main,
    )

    print(
        "\nMoment conditions (full-sample FD estimators):\n"
        f"AB: {ab.n_moments} (excluded instruments: {ab.n_excluded_instr}, "
        f"time dummies: {ab.n_exog})\n"
        f"AH: {ah.n_moments} (excluded instruments: {ah.n_excluded_instr}, "
        f"time dummies: {ah.n_exog})"
    )

    if args.skip_bootstrap:
        bse_fe = np.full(5, np.nan)
        bse_jbc_fe = np.full(5, np.nan)
        bse_lr_fe = np.nan
        bse_lr_jbc = np.nan

        bse_ab = np.full(5, np.nan)
        bse_ab_jbc = np.full(5, np.nan)
        bse_ab_jbc5 = np.full(5, np.nan)
        bse_lr_ab = np.nan
        bse_lr_ab_jbc = np.nan
        bse_lr_ab_jbc5 = np.nan

        bse_ah = np.full(5, np.nan)
        bse_ah_jbc = np.full(5, np.nan)
        bse_ah_jbc5 = np.full(5, np.nan)
        bse_lr_ah = np.nan
        bse_lr_ah_jbc = np.nan
        bse_lr_ah_jbc5 = np.nan

        boot_fe = np.full((0, 18), np.nan)
        boot_fe = np.full((0, 12), np.nan)
        boot_ab = np.full((0, 18), np.nan)
        boot_ah = np.full((0, 18), np.nan)
    else:
        boot_fe = run_bootstrap(
            data=data,
            stat_name="fe",
            reps=args.bootstrap_reps,
            seed=args.seed,
            max_instr_lag=args.max_instr_lag,
        )
        boot_ab = run_bootstrap(
            data=data,
            stat_name="ab",
            reps=args.bootstrap_reps,
            seed=args.seed,
            max_instr_lag=args.max_instr_lag,
        )
        boot_ah = run_bootstrap(
            data=data,
            stat_name="ah",
            reps=args.bootstrap_reps,
            seed=args.seed,
            max_instr_lag=args.max_instr_lag,
        )

        bse_fe = np.array([robust_sd(boot_fe[:, i]) for i in range(0, 5)], dtype=float)
        bse_jbc_fe = np.array(
            [robust_sd(boot_fe[:, i]) for i in range(5, 10)], dtype=float
        )
        bse_lr_fe = robust_sd(boot_fe[:, 10])
        bse_lr_jbc = robust_sd(boot_fe[:, 11])

        bse_ab = np.array([robust_sd(boot_ab[:, i]) for i in range(0, 5)], dtype=float)
        bse_ab_jbc = np.array(
            [robust_sd(boot_ab[:, i]) for i in range(5, 10)], dtype=float
        )
        bse_ab_jbc5 = np.array(
            [robust_sd(boot_ab[:, i]) for i in range(10, 15)], dtype=float
        )
        bse_lr_ab = robust_sd(boot_ab[:, 15])
        bse_lr_ab_jbc = robust_sd(boot_ab[:, 16])
        bse_lr_ab_jbc5 = robust_sd(boot_ab[:, 17])

        bse_ah = np.array([robust_sd(boot_ah[:, i]) for i in range(0, 5)], dtype=float)
        bse_ah_jbc = np.array(
            [robust_sd(boot_ah[:, i]) for i in range(5, 10)], dtype=float
        )
        bse_ah_jbc5 = np.array(
            [robust_sd(boot_ah[:, i]) for i in range(10, 15)], dtype=float
        )
        bse_lr_ah = robust_sd(boot_ah[:, 15])
        bse_lr_ah_jbc = robust_sd(boot_ah[:, 16])
        bse_lr_ah_jbc5 = robust_sd(boot_ah[:, 17])

    table_all = build_table(
        fe=fe,
        coefs_jbc_fe=coefs_jbc_fe,
        lr_jbc_fe=lr_jbc_fe,
        bse_fe=bse_fe,
        bse_jbc_fe=bse_jbc_fe,
        bse_lr_fe=bse_lr_fe,
        bse_lr_jbc=bse_lr_jbc,
        ab=ab,
        coefs_ab_jbc=coefs_ab_jbc,
        coefs_ab_jbc5=coefs_ab_jbc5,
        lr_ab_jbc=lr_ab_jbc,
        lr_ab_jbc5=lr_ab_jbc5,
        bse_ab=bse_ab,
        bse_ab_jbc=bse_ab_jbc,
        bse_ab_jbc5=bse_ab_jbc5,
        bse_lr_ab=bse_lr_ab,
        bse_lr_ab_jbc=bse_lr_ab_jbc,
        bse_lr_ab_jbc5=bse_lr_ab_jbc5,
        ah=ah,
        coefs_ah_jbc=coefs_ah_jbc,
        coefs_ah_jbc5=coefs_ah_jbc5,
        lr_ah_jbc=lr_ah_jbc,
        lr_ah_jbc5=lr_ah_jbc5,
        bse_ah=bse_ah,
        bse_ah_jbc=bse_ah_jbc,
        bse_ah_jbc5=bse_ah_jbc5,
        bse_lr_ah=bse_lr_ah,
        bse_lr_ah_jbc=bse_lr_ah_jbc,
        bse_lr_ah_jbc5=bse_lr_ah_jbc5,
    )

    pd.set_option("display.float_format", lambda x: f"{x:0.2f}")
    print("\nDescriptive statistics:\n")
    print(dstat)
    print("\nTable of results:\n")
    print(table_all)

    results_dir = Path(__file__).resolve().parent / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dstat.to_csv(results_dir / "descriptive_stats.csv")
    table_all.to_csv(results_dir / "table_all.csv")
    write_latex_table(
        table=table_all,
        output_path=results_dir / "table_all.tex",
        caption=args.latex_caption,
        label=args.latex_label,
    )

    payload = {
        "dstat": dstat,
        "fe": fe,
        "coefs_jbc_fe": coefs_jbc_fe,
        "lr_jbc_fe": lr_jbc_fe,
        "ab": ab,
        "coefs_ab_jbc": coefs_ab_jbc,
        "coefs_ab_jbc5": coefs_ab_jbc5,
        "lr_ab_jbc": lr_ab_jbc,
        "lr_ab_jbc5": lr_ab_jbc5,
        "ah": ah,
        "coefs_ah_jbc": coefs_ah_jbc,
        "coefs_ah_jbc5": coefs_ah_jbc5,
        "lr_ah_jbc": lr_ah_jbc,
        "lr_ah_jbc5": lr_ah_jbc5,
        "boot_fe": boot_fe,
        "boot_ab": boot_ab,
        "boot_ah": boot_ah,
        "table_all": table_all,
    }
    with open(results_dir / "democracy_results.pkl", "wb") as f:
        pickle.dump(payload, f)


if __name__ == "__main__":
    main()
