#!/usr/bin/env python3
"""Python translation of Democracy-AER-v2.R.

This script reproduces the main workflow from the original R replication code:
1) Descriptive statistics
2) Two-way fixed effects estimation with clustered SEs
3) Split-panel jackknife bias correction
4) Analytical bias correction
5) Difference-GMM (Arellano-Bond style) using one-step IV-GMM
6) Anderson-Hsiao IV estimation with split-sample bias corrections
7) Panel bootstrap robust standard errors
8) Final summary table

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
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS, IVGMM
from linearmodels.panel import PanelOLS
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

    req = ["d_lgdp", *AB_PARAM_NAMES]
    reg = df.dropna(subset=req).copy()

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

    # Remove collinear all-zero instruments to stabilize GMM in finite samples.
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


def fit_ah(data: pd.DataFrame, max_instr_lag: int) -> EstimationResult:
    # AH uses lagged levels as instruments in the differenced equation.
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

    y = reg["d_lgdp"]
    endog = reg[AB_PARAM_NAMES]
    z = pd.DataFrame(
        {
            "z_dem_l1": reg["dem_l1"],
            "z_y_l2": reg["lgdp_l2"],
            "z_y_l3": reg["lgdp_l3"],
            "z_y_l4": reg["lgdp_l4"],
            "z_y_l5": reg["lgdp_l5"],
        }
    )
    z = z.loc[:, z.var(axis=0) > 0]

    model = IV2SLS(y, exog=None, endog=endog, instruments=z)
    fit = model.fit(cov_type="clustered", clusters=reg["id"].to_numpy())

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
    out = np.full((reps, 18), np.nan, dtype=float)

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

    table[coef_rows, 6] = ah.coefs
    table[cse_rows, 6] = ah.cse
    table[bse_rows, 6] = bse_ah

    table[coef_rows, 7] = coefs_ah_jbc
    table[bse_rows, 7] = bse_ah_jbc

    table[coef_rows, 8] = coefs_ah_jbc5
    table[bse_rows, 8] = bse_ah_jbc5

    table[15, 6] = ah.lr
    table[16, 6] = ah.cse_lr
    table[17, 6] = bse_lr_ah

    table[15, 7] = lr_ah_jbc
    table[17, 7] = bse_lr_ah_jbc

    table[15, 8] = lr_ah_jbc5
    table[17, 8] = bse_lr_ah_jbc5

    table[[0, 1, 2, 15, 16, 17], :] *= 100.0

    return pd.DataFrame(table, index=ROW_NAMES, columns=COL_NAMES)


def write_latex_table(
    table: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
    bootstrap_reps: int,
) -> None:
    def _col(name: str) -> str:
        if name not in table.columns:
            raise KeyError(f"Column '{name}' is required for LaTeX output formatting.")
        return name

    def _to_finite_float(x: object) -> float | None:
        if x is None:
            return None
        if not isinstance(x, (int, float, np.integer, np.floating, str)):
            return None
        try:
            val = float(x)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val):
            return None
        return val

    def _fmt(x: object, wrap: str | None = None) -> str:
        val_num = _to_finite_float(x)
        if val_num is None:
            return ""
        val = f"{val_num:0.2f}"
        if wrap == "()":
            return f"({val})"
        if wrap == "[]":
            return f"[{val}]"
        return val

    def _join(cells: list[str]) -> str:
        return " & ".join(cells)

    def _pick_first_finite(vals: list[object]) -> float:
        for v in vals:
            vv = _to_finite_float(v)
            if vv is not None:
                return vv
        return np.nan

    ah_cols = [_col("AH"), _col("AH-SBC1"), _col("AH-SBC5")]
    ab_cols = [_col("AB"), _col("AB-SBC1"), _col("AB-SBC5")]
    cols = ah_cols + ab_cols

    coef_rows = [
        (r"Democracy ($\times 100$)", 0),
        ("L1.log(gdp)", 3),
        ("L2.log(gdp)", 6),
        ("L3.log(gdp)", 9),
        ("L4.log(gdp)", 12),
    ]

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{AH} & \multicolumn{3}{c}{AB} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    lines.append(r" &  & BC1 & BC5 &  & BC1 & BC5 \\")
    lines.append(r"\midrule")

    for row_label, i in coef_rows:
        coef_vals = [_fmt(table.iloc[i, table.columns.get_loc(c)]) for c in cols]
        cse_vals = [
            _fmt(table.iloc[i + 1, table.columns.get_loc(c)], "()") for c in cols
        ]
        bse_vals = [
            _fmt(table.iloc[i + 2, table.columns.get_loc(c)], "[]") for c in cols
        ]

        lines.append(f"{row_label} & {_join(coef_vals)} \\\\")
        lines.append(f" & {_join(cse_vals)} \\\\")
        lines.append(f" & {_join(bse_vals)} \\\\")
        lines.append("")

    ah_lr = [_fmt(table.iloc[15, table.columns.get_loc(c)]) for c in ah_cols]
    ab_lr = [_fmt(table.iloc[15, table.columns.get_loc(c)]) for c in ab_cols]

    ah_lr_cse = _pick_first_finite(
        [table.iloc[16, table.columns.get_loc(c)] for c in ah_cols]
    )
    ab_lr_cse = _pick_first_finite(
        [table.iloc[16, table.columns.get_loc(c)] for c in ab_cols]
    )
    lr_cse_cells = [_fmt(ah_lr_cse, "()"), "", "", _fmt(ab_lr_cse, "()"), "", ""]

    ah_lr_bse = [_fmt(table.iloc[17, table.columns.get_loc(c)], "[]") for c in ah_cols]
    ab_lr_bse = [_fmt(table.iloc[17, table.columns.get_loc(c)], "[]") for c in ab_cols]

    lines.append(r"\midrule")
    lines.append(
        r"Long-run (democracy) ($\times 100$)" + f" & {_join(ah_lr + ab_lr)} \\\\"
    )
    lines.append(f" & {_join(lr_cse_cells)} \\\\")
    lines.append(f" & {_join(ah_lr_bse + ab_lr_bse)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{0.3em}")
    lines.append(r"\end{table}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        default="Replication results of causal impact of democracy on growth as in Table 10.1.",
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

        bse_ah = np.full(5, np.nan)
        bse_ah_jbc = np.full(5, np.nan)
        bse_ah_jbc5 = np.full(5, np.nan)
        bse_lr_ah = np.nan
        bse_lr_ah_jbc = np.nan
        bse_lr_ah_jbc5 = np.nan

        boot_fe = np.full((0, 18), np.nan)
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
        bootstrap_reps=args.bootstrap_reps,
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
