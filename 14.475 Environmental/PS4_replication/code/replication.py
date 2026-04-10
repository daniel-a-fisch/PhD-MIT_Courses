from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from linearmodels.iv.absorbing import AbsorbingLS


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "Table1.tex"
STATE_OUTPUT_FILE = OUTPUT_DIR / "Table1_state_analogs.tex"
INEQ_OUTPUT_TXT = OUTPUT_DIR / "InequalityAdaptation_interaction.txt"
INEQ_OUTPUT_TEX = OUTPUT_DIR / "InequalityAdaptation_interaction.tex"
LEAD_LAG_FIG = OUTPUT_DIR / "TempLeadLag_29C.png"

MODEL_VARS = ["const", "lower", "higher", "prec_lo", "prec_hi"]
DISPLAY_VARS = ["lower", "higher", "prec_lo", "prec_hi", "const"]
VAR_LABELS = {
    "lower": "GDD below threshold",
    "higher": "GDD above threshold",
    "prec_lo": "Precip below threshold",
    "prec_hi": "Precip above threshold",
    "const": "Constant",
}


@dataclass
class StoredResult:
    params: dict[str, float]
    ses: dict[str, float]
    pvalues: dict[str, float]
    nobs: int
    rsquared: float
    fe_label: str
    t_label: str
    p_label: str
    title: str


def read_dta_as_polars(path: Path) -> pl.DataFrame:
    df = pd.read_stata(path, convert_categoricals=False)
    pl_df = pl.DataFrame(df.to_dict(orient="list"))
    return pl_df.with_columns(
        pl.col(pl.Float64).fill_nan(None),
        pl.col(pl.Float32).fill_nan(None),
    )


def prepare_piecewise(
    df: pl.DataFrame, t_threshold: int, p_threshold: int
) -> pl.DataFrame:
    lower_var = f"dday0_{t_threshold}C_diff1980_2000"
    higher_var = f"dday{t_threshold}C_diff1980_2000"

    p80 = pl.col("prec_smooth1980")
    p00 = pl.col("prec_smooth2000")
    p_cut = pl.lit(float(p_threshold))

    return df.with_columns(
        pl.when((p80 < p_threshold) & (p00 <= p_threshold))
        .then(p00 - p80)
        .when((p80 <= p_threshold) & (p00 > p_threshold))
        .then(p_cut - p80)
        .when((p80 > p_threshold) & (p00 <= p_threshold))
        .then(p00 - p_cut)
        .otherwise(pl.lit(0.0))
        .alias("prec_lo"),
        pl.when((p80 > p_threshold) & (p00 > p_threshold))
        .then(p00 - p80)
        .when((p80 <= p_threshold) & (p00 > p_threshold))
        .then(p00 - p_cut)
        .when((p80 > p_threshold) & (p00 <= p_threshold))
        .then(p_cut - p80)
        .otherwise(pl.lit(0.0))
        .alias("prec_hi"),
        pl.col(lower_var).alias("lower"),
        pl.col(higher_var).alias("higher"),
    ).filter(pl.col("longitude") > -100)


def prepare_panel(df: pl.DataFrame, t_threshold: int, p_threshold: int) -> pl.DataFrame:
    p_cut = pl.lit(float(p_threshold))

    return (
        df.with_columns(
            pl.col("year").cast(pl.Int64, strict=False).alias("year"),
            pl.col("fips").cast(pl.Int64, strict=False).alias("fips"),
        )
        .with_columns(
            pl.col("fips")
            .cast(pl.Utf8)
            .str.zfill(5)
            .str.slice(0, 2)
            .cast(pl.Int64)
            .alias("stfips"),
            pl.when(pl.col("cornyield") > 0)
            .then(pl.col("cornyield").log())
            .otherwise(None)
            .alias("logcornyield"),
            pl.when((pl.col("year") >= 1978) & (pl.col("year") <= 2002))
            .then(pl.col("corn_area"))
            .otherwise(None)
            .mean()
            .over("fips")
            .alias("corn_area_78_02"),
            (pl.col("dday0C") - pl.col(f"dday{t_threshold}C")).alias("lower"),
            pl.col(f"dday{t_threshold}C").alias("higher"),
            pl.when(pl.col("prec").is_null())
            .then(None)
            .when(pl.col("prec") <= p_threshold)
            .then(pl.col("prec") - p_cut)
            .otherwise(pl.lit(0.0))
            .alias("prec_lo"),
            pl.when(pl.col("prec").is_null())
            .then(None)
            .when(pl.col("prec") > p_threshold)
            .then(pl.col("prec") - p_cut)
            .otherwise(pl.lit(0.0))
            .alias("prec_hi"),
        )
        .with_columns(
            (
                pl.col("stfips").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("year").cast(pl.Utf8)
            ).alias("statebyyear")
        )
        .filter(
            (pl.col("longitude") > -100)
            & (pl.col("year") >= 1978)
            & (pl.col("year") <= 2002)
        )
    )


def fit_model(
    df: pl.DataFrame,
    dep_var: str,
    weight_var: str,
    cluster_var: str | None,
    absorb_vars: list[str],
    fe_label: str,
    t_label: str,
    p_label: str,
    title: str,
    *,
    cov_type: str = "clustered",
) -> StoredResult:
    needed = [dep_var, "lower", "higher", "prec_lo", "prec_hi", weight_var]
    if cluster_var is not None:
        needed.append(cluster_var)
    needed.extend(absorb_vars)
    needed = list(dict.fromkeys(needed))

    data = pd.DataFrame(df.select(needed).to_dict(as_series=False))
    data = data.dropna(subset=needed)
    data = data.loc[data[weight_var] > 0].copy()

    y = data[dep_var]
    x = data[["lower", "higher", "prec_lo", "prec_hi"]].copy()
    x.insert(0, "const", 1.0)

    absorb = None
    if absorb_vars:
        absorb = data[absorb_vars].copy()
        for col in absorb_vars:
            absorb[col] = absorb[col].astype("category")

    model = AbsorbingLS(y, x, absorb=absorb, weights=data[weight_var])
    if cov_type == "clustered":
        if cluster_var is None:
            raise ValueError("cluster_var is required when cov_type='clustered'")
        result = model.fit(cov_type="clustered", clusters=data[cluster_var])
    else:
        result = model.fit(cov_type=cov_type)

    return StoredResult(
        params={str(k): float(v) for k, v in result.params.items()},
        ses={str(k): float(v) for k, v in result.std_errors.items()},
        pvalues={str(k): float(v) for k, v in result.pvalues.items()},
        nobs=int(result.nobs),
        rsquared=float(result.rsquared),
        fe_label=fe_label,
        t_label=t_label,
        p_label=p_label,
        title=title,
    )


def stars(pvalue: float) -> str:
    if pvalue < 0.01:
        return r"\sym{***}"
    if pvalue < 0.05:
        return r"\sym{**}"
    if pvalue < 0.10:
        return r"\sym{*}"
    return ""


def fmt_coef(result: StoredResult, var: str) -> str:
    coef = result.params.get(var, float("nan"))
    pvalue = result.pvalues.get(var, 1.0)
    return f"{coef:10.4f}{stars(pvalue)}"


def fmt_se(result: StoredResult, var: str) -> str:
    se = result.ses.get(var, float("nan"))
    return f"({se:0.4f})"


def write_latex_table(results: list[StoredResult], output_file: Path) -> None:
    col_numbers = "&".join(
        [f"\\multicolumn{{1}}{{c}}{{({i})}}" for i in range(1, len(results) + 1)]
    )
    col_titles = "&".join([f"\\multicolumn{{1}}{{c}}{{{r.title}}}" for r in results])

    lines: list[str] = []
    lines.append(r"{\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(
        rf"\begin{{tabular}}{{@{{\hskip\tabcolsep\extracolsep\fill}}l*{{{len(results)}}}{{c}}}}"
    )
    lines.append(r"\hline\hline")
    lines.append(f"                &{col_numbers}\\\\")
    lines.append(f"                &{col_titles}\\\\")
    lines.append(r"\hline")

    for var in DISPLAY_VARS:
        coefs = "&".join([fmt_coef(res, var) for res in results])
        ses = "&".join([fmt_se(res, var) for res in results])
        lines.append(f"{VAR_LABELS[var]}&{coefs}\\\\")
        lines.append(f"                &{ses}\\\\")
        if var != DISPLAY_VARS[-1]:
            lines.append(r"[1em]")

    obs_row = "&".join([f"{res.nobs:10d}" for res in results])
    r2_row = "&".join([f"{res.rsquared:10.3f}" for res in results])
    fe_row = "&".join([f"{res.fe_label:>10}" for res in results])
    t_row = "&".join([f"{res.t_label:>10}" for res in results])
    p_row = "&".join([f"{res.p_label:>10}" for res in results])

    lines.append(r"\hline")
    lines.append(f"Observations    &{obs_row}\\\\")
    lines.append(f"R squared       &{r2_row}\\\\")
    lines.append(f"Fixed Effects   &{fe_row}\\\\")
    lines.append(f"T threshold     &{t_row}\\\\")
    lines.append(f"P threshold     &{p_row}\\\\")
    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}%")
    lines.append("}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")


def _weighted_mean_expr(value_col: str, weight_col: str) -> pl.Expr:
    w = pl.col(weight_col)
    x = pl.col(value_col)
    return (x * w).sum() / w.sum()


def collapse_to_state_longdiff(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse county-level long-difference data to one row per state.

    Uses area-weighted means (weights = corn_area_smooth1980) for y and regressors.
    """

    needed = [
        "stfips",
        "logcornyield_diff1980_2000",
        "lower",
        "higher",
        "prec_lo",
        "prec_hi",
        "corn_area_smooth1980",
    ]
    d = df.select(needed).drop_nulls()
    d = d.filter(pl.col("corn_area_smooth1980") > 0)

    w = "corn_area_smooth1980"
    return (
        d.group_by("stfips")
        .agg(
            pl.col(w).sum().alias("state_corn_area_smooth1980"),
            _weighted_mean_expr("logcornyield_diff1980_2000", w).alias(
                "logcornyield_diff1980_2000"
            ),
            _weighted_mean_expr("lower", w).alias("lower"),
            _weighted_mean_expr("higher", w).alias("higher"),
            _weighted_mean_expr("prec_lo", w).alias("prec_lo"),
            _weighted_mean_expr("prec_hi", w).alias("prec_hi"),
        )
        .sort("stfips")
    )


def collapse_to_state_panel(panel_df: pl.DataFrame) -> pl.DataFrame:
    """Collapse county-year panel data to state-year.

    Uses area-weighted means (weights = corn_area_78_02) for y and regressors.
    """

    needed = [
        "stfips",
        "year",
        "logcornyield",
        "lower",
        "higher",
        "prec_lo",
        "prec_hi",
        "corn_area_78_02",
    ]
    d = panel_df.select(needed).drop_nulls()
    d = d.filter(pl.col("corn_area_78_02") > 0)

    w = "corn_area_78_02"
    return (
        d.group_by(["stfips", "year"])
        .agg(
            pl.col(w).sum().alias("state_corn_area_78_02"),
            _weighted_mean_expr("logcornyield", w).alias("logcornyield"),
            _weighted_mean_expr("lower", w).alias("lower"),
            _weighted_mean_expr("higher", w).alias("higher"),
            _weighted_mean_expr("prec_lo", w).alias("prec_lo"),
            _weighted_mean_expr("prec_hi", w).alias("prec_hi"),
        )
        .sort(["stfips", "year"])
    )


def run_state_level_analogs(
    *,
    piecewise_raw: pl.DataFrame,
    panel_raw: pl.DataFrame,
    t_threshold: int = 29,
    p_threshold: int = 42,
) -> list[StoredResult]:
    """Run state-level analogs of Table 1 columns (1) and (3)."""

    table_t = f"{t_threshold}C"
    table_p = f"{p_threshold}cm"

    diff_df = prepare_piecewise(piecewise_raw, t_threshold, p_threshold)
    state_diff = collapse_to_state_longdiff(diff_df)

    # With only ~48 states, clustered-by-state SEs are not meaningful here;
    # use heteroskedasticity-robust SEs.
    res_diff = fit_model(
        df=state_diff,
        dep_var="logcornyield_diff1980_2000",
        weight_var="state_corn_area_smooth1980",
        cluster_var=None,
        absorb_vars=[],
        fe_label="None",
        t_label=table_t,
        p_label=table_p,
        title="State Diffs",
        cov_type="robust",
    )

    panel_df = prepare_panel(panel_raw, t_threshold, p_threshold)
    state_panel = collapse_to_state_panel(panel_df)
    res_panel = fit_model(
        df=state_panel,
        dep_var="logcornyield",
        weight_var="state_corn_area_78_02",
        cluster_var="stfips",
        absorb_vars=["stfips", "year"],
        fe_label="State, Yr",
        t_label=table_t,
        p_label=table_p,
        title="State Panel",
        cov_type="clustered",
    )

    return [res_diff, res_panel]


def fit_inequality_adaptation_interaction(
    panel_raw: pl.DataFrame,
    *,
    t_threshold: int = 29,
    p_threshold: int = 42,
    baseline_start: int = 1978,
    baseline_end: int = 1982,
    out_txt: Path = INEQ_OUTPUT_TXT,
    out_tex: Path = INEQ_OUTPUT_TEX,
) -> None:
    """Test whether baseline productivity attenuates heat damages.

    Estimating equation (county-year panel):
      log(y_it) = a_i + g_t + b1*lower_it + b2*higher_it + b3*prec_lo_it + b4*prec_hi_it
                 + theta*(higher_it * Prod_i) + eps_it

    where Prod_i is predetermined baseline productivity (mean log yield over baseline years).
    """

    df = prepare_panel(panel_raw, t_threshold=t_threshold, p_threshold=p_threshold)

    baseline = (
        df.filter((pl.col("year") >= baseline_start) & (pl.col("year") <= baseline_end))
        .group_by("fips")
        .agg(pl.col("logcornyield").mean().alias("baseline_logcornyield"))
    )

    df = df.join(baseline, on="fips", how="left")

    baseline_stats = df.select(pl.col("baseline_logcornyield")).drop_nulls()
    baseline_mean = float(
        baseline_stats.select(pl.col("baseline_logcornyield").mean()).item()
    )
    baseline_std = float(
        baseline_stats.select(pl.col("baseline_logcornyield").std()).item()
    )
    if not (baseline_std > 0):
        raise ValueError(
            "baseline productivity std is non-positive; cannot standardize"
        )

    df = df.with_columns(
        (
            (pl.col("baseline_logcornyield") - pl.lit(baseline_mean))
            / pl.lit(baseline_std)
        ).alias("baseline_logcornyield_z")
    )
    df = df.with_columns(
        (pl.col("higher") * pl.col("baseline_logcornyield_z")).alias(
            "higher_x_baseline_prod"
        )
    )

    needed = [
        "logcornyield",
        "lower",
        "higher",
        "higher_x_baseline_prod",
        "prec_lo",
        "prec_hi",
        "corn_area_78_02",
        "stfips",
        "fips",
        "year",
    ]
    data = pd.DataFrame(df.select(needed).to_dict(as_series=False))
    data = data.dropna(subset=needed)
    data = data.loc[data["corn_area_78_02"] > 0].copy()

    y = data["logcornyield"]
    x = data[["lower", "higher", "higher_x_baseline_prod", "prec_lo", "prec_hi"]].copy()
    x.insert(0, "const", 1.0)

    absorb = data[["fips", "year"]].copy()
    absorb["fips"] = absorb["fips"].astype("category")
    absorb["year"] = absorb["year"].astype("category")

    model = AbsorbingLS(y, x, absorb=absorb, weights=data["corn_area_78_02"])
    res = model.fit(cov_type="clustered", clusters=data["stfips"])

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(res.summary.as_text(), encoding="utf-8")

    try:
        out_tex.write_text(res.summary.as_latex(), encoding="utf-8")
    except Exception:
        # Some linearmodels versions may not support LaTeX export.
        pass

    print(f"Wrote {out_txt}")
    if out_tex.exists():
        print(f"Wrote {out_tex}")


def _make_lead_lag_names(
    leads: int, lags: int, omit: int | None
) -> list[tuple[int, str]]:
    terms: list[tuple[int, str]] = []
    for k in range(-leads, lags + 1):
        if omit is not None and k == omit:
            continue
        if k < 0:
            name = f"higher_km{abs(k)}"
        elif k > 0:
            name = f"higher_kp{k}"
        else:
            name = "higher_k0"
        terms.append((k, name))
    return terms


def fit_temperature_lead_lag_and_plot(
    panel_raw: pl.DataFrame,
    *,
    t_threshold: int = 29,
    p_threshold: int = 42,
    leads: int = 5,
    lags: int = 5,
    omit: int | None = -1,
    fig_path: Path = LEAD_LAG_FIG,
) -> None:
    """Estimate a distributed lead/lag regression for extreme heat and save a plot.

    Spec (conceptually):
      logcornyield_it = alpha_i + gamma_t + sum_k beta_k * higher_{i,t-k}
                      + controls_it + eps_it

    - Absorb: county (fips) and year fixed effects
    - Cluster: state (stfips)
    - Weights: county mean corn area over 1978-2002
    """

    terms = _make_lead_lag_names(leads=leads, lags=lags, omit=omit)
    p_cut = pl.lit(float(p_threshold))

    df = (
        panel_raw.with_columns(
            pl.col("year").cast(pl.Int64, strict=False).alias("year"),
            pl.col("fips").cast(pl.Int64, strict=False).alias("fips"),
        )
        .with_columns(
            pl.col("fips")
            .cast(pl.Utf8)
            .str.zfill(5)
            .str.slice(0, 2)
            .cast(pl.Int64)
            .alias("stfips"),
            pl.when(pl.col("cornyield") > 0)
            .then(pl.col("cornyield").log())
            .otherwise(None)
            .alias("logcornyield"),
            pl.when((pl.col("year") >= 1978) & (pl.col("year") <= 2002))
            .then(pl.col("corn_area"))
            .otherwise(None)
            .mean()
            .over("fips")
            .alias("corn_area_78_02"),
            (pl.col("dday0C") - pl.col(f"dday{t_threshold}C")).alias("lower"),
            pl.col(f"dday{t_threshold}C").alias("higher"),
            pl.when(pl.col("prec").is_null())
            .then(None)
            .when(pl.col("prec") <= p_threshold)
            .then(pl.col("prec") - p_cut)
            .otherwise(pl.lit(0.0))
            .alias("prec_lo"),
            pl.when(pl.col("prec").is_null())
            .then(None)
            .when(pl.col("prec") > p_threshold)
            .then(pl.col("prec") - p_cut)
            .otherwise(pl.lit(0.0))
            .alias("prec_hi"),
        )
        .sort(["fips", "year"], nulls_last=True)
    )

    lag_exprs = [
        pl.col("higher").shift(k).over("fips").alias(name) for k, name in terms
    ]
    df = df.with_columns(*lag_exprs)

    df = df.filter(
        (pl.col("longitude") > -100)
        & (pl.col("year") >= 1978)
        & (pl.col("year") <= 2002)
    )

    exog_vars = [name for _, name in terms]
    needed = [
        "logcornyield",
        "lower",
        "prec_lo",
        "prec_hi",
        "corn_area_78_02",
        "stfips",
        "fips",
        "year",
    ] + exog_vars
    needed = list(dict.fromkeys(needed))

    data = pd.DataFrame(df.select(needed).to_dict(as_series=False))
    data = data.dropna(subset=needed)
    data = data.loc[data["corn_area_78_02"] > 0].copy()

    y = data["logcornyield"]
    x = data[["lower", "prec_lo", "prec_hi"] + exog_vars].copy()
    x.insert(0, "const", 1.0)
    absorb = data[["fips", "year"]].copy()
    absorb["fips"] = absorb["fips"].astype("category")
    absorb["year"] = absorb["year"].astype("category")

    model = AbsorbingLS(y, x, absorb=absorb, weights=data["corn_area_78_02"])
    res = model.fit(cov_type="clustered", clusters=data["stfips"])

    ks = [k for k, _ in terms]
    betas = [float(res.params[name]) for _, name in terms]
    ses = [float(res.std_errors[name]) for _, name in terms]
    ci_low = [b - 1.96 * s for b, s in zip(betas, ses)]
    ci_high = [b + 1.96 * s for b, s in zip(betas, ses)]

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.plot(ks, betas, marker="o", linewidth=1.5)
    plt.fill_between(ks, ci_low, ci_high, alpha=0.2)
    if omit is not None:
        plt.axvline(omit, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("year leads/lags (k)")
    plt.ylabel("Coefficient on extreme heat")
    # plt.title(f"Distributed lead/lag: dday{t_threshold}C on log corn yield")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Wrote {fig_path}")


def main() -> None:
    piecewise = read_dta_as_polars(DATA_DIR / "yield_piecewise.dta")
    panel = read_dta_as_polars(DATA_DIR / "us_panel.dta")

    all_results: list[StoredResult] = []

    for t_threshold, p_threshold in [(29, 42), (28, 50)]:
        table_t = f"{t_threshold}C"
        table_p = f"{p_threshold}cm"

        diff_df = prepare_piecewise(piecewise, t_threshold, p_threshold)
        all_results.append(
            fit_model(
                df=diff_df,
                dep_var="logcornyield_diff1980_2000",
                weight_var="corn_area_smooth1980",
                cluster_var="stfips",
                absorb_vars=[],
                fe_label="None",
                t_label=table_t,
                p_label=table_p,
                title="Diffs",
            )
        )
        all_results.append(
            fit_model(
                df=diff_df,
                dep_var="logcornyield_diff1980_2000",
                weight_var="corn_area_smooth1980",
                cluster_var="stfips",
                absorb_vars=["stfips"],
                fe_label="State",
                t_label=table_t,
                p_label=table_p,
                title="Diffs",
            )
        )

        panel_df = prepare_panel(panel, t_threshold, p_threshold)
        all_results.append(
            fit_model(
                df=panel_df,
                dep_var="logcornyield",
                weight_var="corn_area_78_02",
                cluster_var="stfips",
                absorb_vars=["fips", "year"],
                fe_label="Cty, Yr",
                t_label=table_t,
                p_label=table_p,
                title="Panel",
            )
        )
        all_results.append(
            fit_model(
                df=panel_df,
                dep_var="logcornyield",
                weight_var="corn_area_78_02",
                cluster_var="stfips",
                absorb_vars=["fips", "statebyyear"],
                fe_label="Cty, State-Yr",
                t_label=table_t,
                p_label=table_p,
                title="Panel",
            )
        )

    write_latex_table(all_results, OUTPUT_FILE)
    print(f"Wrote {OUTPUT_FILE}")

    state_results = run_state_level_analogs(
        piecewise_raw=piecewise,
        panel_raw=panel,
        t_threshold=29,
        p_threshold=42,
    )
    write_latex_table(state_results, STATE_OUTPUT_FILE)
    print(f"Wrote {STATE_OUTPUT_FILE}")

    fit_inequality_adaptation_interaction(
        panel,
        t_threshold=29,
        p_threshold=42,
        baseline_start=1978,
        baseline_end=1982,
    )

    fit_temperature_lead_lag_and_plot(panel, t_threshold=29, p_threshold=42)


if __name__ == "__main__":
    main()
