#!/usr/bin/env python3
"""Baseline Python replication of Democracy-AER-v2.R.

This script keeps the original R-style estimator set:
1) Descriptive statistics
2) Two-way FE with clustered SE
3) FE split-panel jackknife correction (SBC)
4) FE analytical bias correction with lag trim 4 (ABC4)
5) AB-style difference GMM
6) SH-style score moments estimator
7) AB/SH split-sample bias corrections (SBC1, SBC5)
8) Panel bootstrap robust SE
9) Paper-style LaTeX table output

Outputs are written under Results/ using *_base filenames.
"""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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
    "ABC4",
    "AB",
    "AB-SBC1",
    "AB-SBC5",
    "SH",
    "SH-SBC1",
    "SH-SBC5",
]


@dataclass
class EstimationResult:
    coefs: np.ndarray
    cov: np.ndarray
    cse: np.ndarray
    lr: float
    cse_lr: float


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
    reg = df.dropna(subset=["lgdp", *FE_PARAM_NAMES])

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


def abc_bias(data: pd.DataFrame, lags: int = 4) -> np.ndarray:
    df = add_lags(data, "lgdp", 4)
    df = df.rename(
        columns={
            "lgdp_l1": "l1lgdp",
            "lgdp_l2": "l2lgdp",
            "lgdp_l3": "l3lgdp",
            "lgdp_l4": "l4lgdp",
        }
    )
    df = df.dropna(subset=["lgdp", "dem", "l1lgdp", "l2lgdp", "l3lgdp", "l4lgdp"])
    df = df.sort_values(["id", "year"]).reset_index(drop=True)

    fit = smf.ols(
        "lgdp ~ dem + l1lgdp + l2lgdp + l3lgdp + l4lgdp + C(year) + C(id)",
        data=df,
    ).fit()

    x = np.asarray(fit.model.exog, dtype=float)
    res = np.asarray(fit.resid, dtype=float)
    n = res.shape[0]
    n_entities = df["id"].nunique()
    t_obs = n // n_entities

    jac = np.linalg.pinv((x.T @ x) / n)[1:6, 1:6]

    indexes = np.arange(n)
    bscore = np.zeros(5, dtype=float)
    for i in range(1, lags + 1):
        drop_pos = np.arange(0, n_entities * t_obs, t_obs)
        drop_pos = drop_pos[drop_pos < len(indexes)]
        indexes = np.delete(indexes, drop_pos)
        lagged_indexes = indexes - i
        bscore += x[indexes, 1:6].T @ res[lagged_indexes] / len(indexes)

    bias = -jac @ bscore
    return np.asarray(bias / t_obs, dtype=float)


def fit_ab(data: pd.DataFrame, max_instr_lag: int) -> EstimationResult:
    df = add_lags(data, "lgdp", max(max_instr_lag, 5))
    df = add_lags(df, "dem", max_instr_lag)

    grp = df.groupby("id", sort=False)
    df["d_lgdp"] = grp["lgdp"].diff()
    df["d_dem"] = grp["dem"].diff()
    for k in range(1, 5):
        df[f"d_lgdp_l{k}"] = df[f"lgdp_l{k}"] - df[f"lgdp_l{k+1}"]

    reg = df.dropna(subset=["d_lgdp", *AB_PARAM_NAMES]).copy()

    for j in range(2, max_instr_lag + 1):
        reg[f"z_y_l{j}"] = reg[f"lgdp_l{j}"].fillna(0.0)
    for j in range(1, max_instr_lag + 1):
        reg[f"z_dem_l{j}"] = reg[f"dem_l{j}"].fillna(0.0)

    y = reg["d_lgdp"]
    endog = reg[AB_PARAM_NAMES]
    z_cols = [
        c for c in reg.columns if c.startswith("z_y_l") or c.startswith("z_dem_l")
    ]
    z = reg[z_cols]
    z = z.loc[:, z.var(axis=0) > 0]

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

    return EstimationResult(coefs=coefs, cov=cov, cse=cse, lr=lr, cse_lr=cse_lr)


def fit_sh(data: pd.DataFrame, max_instr_lag: int) -> EstimationResult:
    # SH score moments use lagged structural regressors in period-specific moment stacks.
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

    return EstimationResult(coefs=coefs, cov=cov, cse=cse, lr=lr, cse_lr=cse_lr)


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


def split_sample_sh_correction(
    data: pd.DataFrame,
    base_sh: EstimationResult,
    partitions: int,
    max_instr_lag: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    ids = np.array(sorted(data["id"].unique()))
    n = len(ids)

    avg_coefs = np.zeros_like(base_sh.coefs)
    avg_lr = 0.0

    for _ in range(partitions):
        sample1 = set(
            rng.choice(ids, size=int(math.ceil(n / 2.0)), replace=False).tolist()
        )
        d1 = data[data["id"].isin(sample1)].copy()
        d2 = data[~data["id"].isin(sample1)].copy()

        sh1 = fit_sh(d1, max_instr_lag=max_instr_lag)
        sh2 = fit_sh(d2, max_instr_lag=max_instr_lag)

        avg_coefs += 0.5 * (sh1.coefs + sh2.coefs) / partitions
        avg_lr += 0.5 * (sh1.lr + sh2.lr) / partitions

    coefs_jbc = 2.0 * base_sh.coefs - avg_coefs
    lr_jbc = 2.0 * base_sh.lr - avg_lr
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

        bias_l4 = abc_bias(data, lags=4)
        coefs_abc4 = fe.coefs - bias_l4

        denom = 1.0 - np.sum(fe.coefs[1:5])
        jac_lr = np.array([1.0, fe.lr, fe.lr, fe.lr, fe.lr], dtype=float) / denom
        lr_abc4 = float(fe.lr - jac_lr @ bias_l4)

        return np.concatenate(
            [
                fe.coefs,
                coefs_jbc,
                coefs_abc4,
                np.array([fe.lr, lr_jbc, lr_abc4], dtype=float),
            ]
        )
    except Exception:
        return np.full(18, np.nan, dtype=float)


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


def boot_stat_sh(
    data: pd.DataFrame, rng: np.random.Generator, max_instr_lag: int
) -> np.ndarray:
    try:
        sh = fit_sh(data, max_instr_lag=max_instr_lag)
        coefs_jbc, lr_jbc = split_sample_sh_correction(
            data,
            base_sh=sh,
            partitions=1,
            max_instr_lag=max_instr_lag,
            rng=rng,
        )
        coefs_jbc5, lr_jbc5 = split_sample_sh_correction(
            data,
            base_sh=sh,
            partitions=5,
            max_instr_lag=max_instr_lag,
            rng=rng,
        )
        return np.concatenate(
            [
                sh.coefs,
                coefs_jbc,
                coefs_jbc5,
                np.array([sh.lr, lr_jbc, lr_jbc5], dtype=float),
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
    out = np.full((reps, 18), np.nan, dtype=float)

    for r in range(reps):
        rng = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
        sample = bootstrap_resample(data, rng)
        if stat_name == "fe":
            out[r, :] = boot_stat_fe(sample, rng)
        elif stat_name == "ab":
            out[r, :] = boot_stat_ab(sample, rng, max_instr_lag=max_instr_lag)
        elif stat_name == "sh":
            out[r, :] = boot_stat_sh(sample, rng, max_instr_lag=max_instr_lag)
        else:
            raise ValueError(f"Unknown bootstrap statistic: {stat_name}")

    return out


def build_table(
    fe: EstimationResult,
    coefs_jbc_fe: np.ndarray,
    coefs_abc4: np.ndarray,
    lr_jbc_fe: float,
    lr_abc4: float,
    bse_fe: np.ndarray,
    bse_jbc_fe: np.ndarray,
    bse_abc4: np.ndarray,
    bse_lr_fe: float,
    bse_lr_jbc: float,
    bse_lr_abc4: float,
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
    sh: EstimationResult,
    coefs_sh_jbc: np.ndarray,
    coefs_sh_jbc5: np.ndarray,
    lr_sh_jbc: float,
    lr_sh_jbc5: float,
    bse_sh: np.ndarray,
    bse_sh_jbc: np.ndarray,
    bse_sh_jbc5: np.ndarray,
    bse_lr_sh: float,
    bse_lr_sh_jbc: float,
    bse_lr_sh_jbc5: float,
) -> pd.DataFrame:
    table = np.full((18, 9), np.nan, dtype=float)

    coef_rows = [0, 3, 6, 9, 12]
    cse_rows = [1, 4, 7, 10, 13]
    bse_rows = [2, 5, 8, 11, 14]

    table[coef_rows, 0] = fe.coefs
    table[cse_rows, 0] = fe.cse
    table[bse_rows, 0] = bse_fe

    table[coef_rows, 1] = coefs_jbc_fe
    table[bse_rows, 1] = bse_jbc_fe

    table[coef_rows, 2] = coefs_abc4
    table[bse_rows, 2] = bse_abc4

    table[coef_rows, 3] = ab.coefs
    table[cse_rows, 3] = ab.cse
    table[bse_rows, 3] = bse_ab

    table[coef_rows, 4] = coefs_ab_jbc
    table[bse_rows, 4] = bse_ab_jbc

    table[coef_rows, 5] = coefs_ab_jbc5
    table[bse_rows, 5] = bse_ab_jbc5

    table[coef_rows, 6] = sh.coefs
    table[cse_rows, 6] = sh.cse
    table[bse_rows, 6] = bse_sh

    table[coef_rows, 7] = coefs_sh_jbc
    table[bse_rows, 7] = bse_sh_jbc

    table[coef_rows, 8] = coefs_sh_jbc5
    table[bse_rows, 8] = bse_sh_jbc5

    table[15, 0] = fe.lr
    table[16, 0] = fe.cse_lr
    table[17, 0] = bse_lr_fe

    table[15, 1] = lr_jbc_fe
    table[17, 1] = bse_lr_jbc

    table[15, 2] = lr_abc4
    table[17, 2] = bse_lr_abc4

    table[15, 3] = ab.lr
    table[17, 3] = bse_lr_ab

    table[15, 4] = lr_ab_jbc
    table[16, 4] = ab.cse_lr
    table[17, 4] = bse_lr_ab_jbc

    table[15, 5] = lr_ab_jbc5
    table[17, 5] = bse_lr_ab_jbc5

    table[15, 6] = sh.lr
    table[16, 6] = sh.cse_lr
    table[17, 6] = bse_lr_sh

    table[15, 7] = lr_sh_jbc
    table[17, 7] = bse_lr_sh_jbc

    table[15, 8] = lr_sh_jbc5
    table[17, 8] = bse_lr_sh_jbc5

    table[[0, 1, 2, 15, 16, 17], :] *= 100.0

    return pd.DataFrame(table, index=ROW_NAMES, columns=COL_NAMES)


def _fmt(v: object) -> str:
    if pd.isna(cast(Any, v)):
        return ""
    return f"{v:0.2f}"


def _wrap(v: object, left: str, right: str) -> str:
    if pd.isna(cast(Any, v)):
        return ""
    return f"{left}{v:0.2f}{right}"


def write_latex_table(
    table: pd.DataFrame, output_path: Path, caption: str, label: str
) -> None:
    cols = [
        "FE",
        "SBC",
        "ABC4",
        "AB",
        "AB-SBC1",
        "AB-SBC5",
        "SH",
        "SH-SBC1",
        "SH-SBC5",
    ]
    blocks = [
        ("Democracy (x100)", "Democracy", "CSE", "BSE"),
        ("L1.log(gdp)", "L1.log(gdp)", "CSE1", "BSE1"),
        ("L2.log(gdp)", "L2.log(gdp)", "CSE2", "BSE2"),
        ("L3.log(gdp)", "L3.log(gdp)", "CSE3", "BSE3"),
        ("L4.log(gdp)", "L4.log(gdp)", "CSE4", "BSE4"),
        ("Long-run democracy (x100)", "LR-Democracy", "CSE5", "BSE5"),
    ]

    lines: list[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lrrrrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        " & \\multicolumn{3}{c}{FE-based} & \\multicolumn{3}{c}{AB-based} & \\multicolumn{3}{c}{SH-based} \\\\"
    )
    lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}")
    lines.append(" & FE & SBC & ABC4 & AB & BC1 & BC5 & SH & BC1 & BC5 \\\\")
    lines.append("\\midrule")

    for i, (label_row, coef_row, cse_row, bse_row) in enumerate(blocks):
        coef_vals = " & ".join(_fmt(table.loc[coef_row, c]) for c in cols)
        cse_vals = " & ".join(_wrap(table.loc[cse_row, c], "(", ")") for c in cols)
        bse_vals = " & ".join(_wrap(table.loc[bse_row, c], "[", "]") for c in cols)

        lines.append(f"{label_row} & {coef_vals} \\\\")
        lines.append(f" & {cse_vals} \\\\")
        lines.append(f" & {bse_vals} \\\\")
        if i < len(blocks) - 1:
            lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.5em}")
    lines.append("\\begin{flushleft}")
    lines.append("\\footnotesize")
    lines.append(
        "Note 1: FE includes two-way effects; AB and SH follow one-step difference-GMM baseline replication variants.\\\\"
    )
    lines.append(
        "Note 2: Clustered standard errors (when reported) are at the country level in parentheses.\\\\"
    )
    lines.append(
        "Note 3: Bootstrap standard errors are in brackets and based on panel resampling by country."
    )
    lines.append("\\end{flushleft}")
    lines.append("\\end{table}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline replication of Democracy-AER-v2.R in Python"
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
        help="Number of panel bootstrap replications.",
    )
    parser.add_argument(
        "--max-instr-lag",
        type=int,
        default=22,
        help="Maximum lag depth for AB/SH instrument construction.",
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
        default="Baseline replication results: democracy and growth",
        help="Caption used in the LaTeX table.",
    )
    parser.add_argument(
        "--latex-label",
        type=str,
        default="tab:democracy-replication-base",
        help="Label used in the LaTeX table.",
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

    bias_l4 = abc_bias(data, lags=4)
    coefs_abc4 = fe.coefs - bias_l4
    denom_fe = 1.0 - np.sum(fe.coefs[1:5])
    jac_lr_fe = np.array([1.0, fe.lr, fe.lr, fe.lr, fe.lr], dtype=float) / denom_fe
    lr_abc4 = float(fe.lr - jac_lr_fe @ bias_l4)

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

    sh = fit_sh(data, max_instr_lag=args.max_instr_lag)
    coefs_sh_jbc, lr_sh_jbc = split_sample_sh_correction(
        data,
        base_sh=sh,
        partitions=1,
        max_instr_lag=args.max_instr_lag,
        rng=rng_main,
    )
    coefs_sh_jbc5, lr_sh_jbc5 = split_sample_sh_correction(
        data,
        base_sh=sh,
        partitions=5,
        max_instr_lag=args.max_instr_lag,
        rng=rng_main,
    )

    if args.skip_bootstrap:
        bse_fe = np.full(5, np.nan)
        bse_jbc_fe = np.full(5, np.nan)
        bse_abc4 = np.full(5, np.nan)
        bse_lr_fe = np.nan
        bse_lr_jbc = np.nan
        bse_lr_abc4 = np.nan

        bse_ab = np.full(5, np.nan)
        bse_ab_jbc = np.full(5, np.nan)
        bse_ab_jbc5 = np.full(5, np.nan)
        bse_lr_ab = np.nan
        bse_lr_ab_jbc = np.nan
        bse_lr_ab_jbc5 = np.nan

        bse_sh = np.full(5, np.nan)
        bse_sh_jbc = np.full(5, np.nan)
        bse_sh_jbc5 = np.full(5, np.nan)
        bse_lr_sh = np.nan
        bse_lr_sh_jbc = np.nan
        bse_lr_sh_jbc5 = np.nan

        boot_fe = np.full((0, 18), np.nan)
        boot_ab = np.full((0, 18), np.nan)
        boot_sh = np.full((0, 18), np.nan)
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
        boot_sh = run_bootstrap(
            data=data,
            stat_name="sh",
            reps=args.bootstrap_reps,
            seed=args.seed,
            max_instr_lag=args.max_instr_lag,
        )

        bse_fe = np.array([robust_sd(boot_fe[:, i]) for i in range(0, 5)], dtype=float)
        bse_jbc_fe = np.array(
            [robust_sd(boot_fe[:, i]) for i in range(5, 10)], dtype=float
        )
        bse_abc4 = np.array(
            [robust_sd(boot_fe[:, i]) for i in range(10, 15)], dtype=float
        )
        bse_lr_fe = robust_sd(boot_fe[:, 15])
        bse_lr_jbc = robust_sd(boot_fe[:, 16])
        bse_lr_abc4 = robust_sd(boot_fe[:, 17])

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

        bse_sh = np.array([robust_sd(boot_sh[:, i]) for i in range(0, 5)], dtype=float)
        bse_sh_jbc = np.array(
            [robust_sd(boot_sh[:, i]) for i in range(5, 10)], dtype=float
        )
        bse_sh_jbc5 = np.array(
            [robust_sd(boot_sh[:, i]) for i in range(10, 15)], dtype=float
        )
        bse_lr_sh = robust_sd(boot_sh[:, 15])
        bse_lr_sh_jbc = robust_sd(boot_sh[:, 16])
        bse_lr_sh_jbc5 = robust_sd(boot_sh[:, 17])

    table_base = build_table(
        fe=fe,
        coefs_jbc_fe=coefs_jbc_fe,
        coefs_abc4=coefs_abc4,
        lr_jbc_fe=lr_jbc_fe,
        lr_abc4=lr_abc4,
        bse_fe=bse_fe,
        bse_jbc_fe=bse_jbc_fe,
        bse_abc4=bse_abc4,
        bse_lr_fe=bse_lr_fe,
        bse_lr_jbc=bse_lr_jbc,
        bse_lr_abc4=bse_lr_abc4,
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
        sh=sh,
        coefs_sh_jbc=coefs_sh_jbc,
        coefs_sh_jbc5=coefs_sh_jbc5,
        lr_sh_jbc=lr_sh_jbc,
        lr_sh_jbc5=lr_sh_jbc5,
        bse_sh=bse_sh,
        bse_sh_jbc=bse_sh_jbc,
        bse_sh_jbc5=bse_sh_jbc5,
        bse_lr_sh=bse_lr_sh,
        bse_lr_sh_jbc=bse_lr_sh_jbc,
        bse_lr_sh_jbc5=bse_lr_sh_jbc5,
    )

    pd.set_option("display.float_format", lambda x: f"{x:0.2f}")
    print("\nDescriptive statistics:\n")
    print(dstat)
    print("\nBaseline table of results:\n")
    print(table_base)

    results_dir = Path(__file__).resolve().parent / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dstat.to_csv(results_dir / "descriptive_stats_base.csv")
    table_base.to_csv(results_dir / "table_base.csv")
    write_latex_table(
        table=table_base,
        output_path=results_dir / "table_base.tex",
        caption=args.latex_caption,
        label=args.latex_label,
    )

    payload = {
        "dstat": dstat,
        "fe": fe,
        "coefs_jbc_fe": coefs_jbc_fe,
        "coefs_abc4": coefs_abc4,
        "lr_jbc_fe": lr_jbc_fe,
        "lr_abc4": lr_abc4,
        "ab": ab,
        "coefs_ab_jbc": coefs_ab_jbc,
        "coefs_ab_jbc5": coefs_ab_jbc5,
        "lr_ab_jbc": lr_ab_jbc,
        "lr_ab_jbc5": lr_ab_jbc5,
        "sh": sh,
        "coefs_sh_jbc": coefs_sh_jbc,
        "coefs_sh_jbc5": coefs_sh_jbc5,
        "lr_sh_jbc": lr_sh_jbc,
        "lr_sh_jbc5": lr_sh_jbc5,
        "boot_fe": boot_fe,
        "boot_ab": boot_ab,
        "boot_sh": boot_sh,
        "table_base": table_base,
    }
    with open(results_dir / "democracy_results_base.pkl", "wb") as f:
        pickle.dump(payload, f)


if __name__ == "__main__":
    main()
