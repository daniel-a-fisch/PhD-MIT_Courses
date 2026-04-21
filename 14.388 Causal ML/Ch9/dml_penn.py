"""Double ML illustration for the Penn reemployment experiment.

This script mirrors the PLM empirical analysis and applies cross-fitted PLR
and extends it also to IRM estimators to treatment group 4 vs control in the
Pennsylvania reemployment bonus RCT.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from patsy import dmatrix  # type: ignore[attr-defined]
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(1234)


def qr_decomposition(x: pd.DataFrame, threshold: float = 1e-6) -> pd.DataFrame:
    """Drop (near) collinear columns using QR decomposition."""
    # Keep only columns with sufficiently large diagonal entries in R.
    # This stabilizes downstream ML fits when the design matrix is high-dimensional.
    _, rx = np.linalg.qr(x)
    keep = np.where(np.abs(np.diag(rx)) > threshold)[0]
    return x.iloc[:, keep]


def dml_plr(
    x: pd.DataFrame,
    d: np.ndarray,
    y: np.ndarray,
    model_y,
    model_d,
    *,
    nfolds: int,
    classifier: bool = False,
):
    """Cross-fitted DML for the partially linear regression model."""
    # Build folds once and reuse across nuisance predictions for comparability.
    cv = KFold(n_splits=nfolds, shuffle=True, random_state=123)

    # Step 1: obtain out-of-fold nuisance predictions for E[Y|X] and E[D|X].
    y_hat = cross_val_predict(model_y, x, y, cv=cv, n_jobs=-1)
    if classifier:
        d_hat = cross_val_predict(
            model_d, x, d, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
    else:
        d_hat = cross_val_predict(model_d, x, d, cv=cv, n_jobs=-1)

    # Step 2: orthogonalize outcome and treatment via residualization.
    y_res = y - y_hat
    d_res = d - d_hat

    # Step 3: final residual-on-residual moment estimate and asymptotic SE.
    theta_hat = np.mean(y_res * d_res) / np.mean(d_res**2)
    eps = y_res - theta_hat * d_res
    var_hat = np.mean(eps**2 * d_res**2) / np.mean(d_res**2) ** 2
    se_hat = np.sqrt(var_hat / x.shape[0])

    return float(theta_hat), float(se_hat), y_res, d_res


def dml_irm(
    x: pd.DataFrame,
    d: np.ndarray,
    y: np.ndarray,
    model_y0,
    model_y1,
    model_d,
    *,
    trimming: float = 0.01,
    nfolds: int,
):
    """Cross-fitted doubly robust DML for the interactive regression model."""
    # Reuse a fixed split for all nuisance components.
    cv = KFold(n_splits=nfolds, shuffle=True, random_state=123)

    y_hat0 = np.zeros_like(y, dtype=float)
    y_hat1 = np.zeros_like(y, dtype=float)

    # Step 1: estimate separate outcome models for untreated and treated groups.
    for train_idx, test_idx in cv.split(x, y):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train = y[train_idx]
        d_train = d[train_idx]

        y_hat0[test_idx] = (
            clone(model_y0)
            .fit(x_train[d_train == 0], y_train[d_train == 0])
            .predict(x_test)
        )
        y_hat1[test_idx] = (
            clone(model_y1)
            .fit(x_train[d_train == 1], y_train[d_train == 1])
            .predict(x_test)
        )

    # Step 2: estimate and trim propensity scores.
    y_hat_obs = y_hat0 * (1 - d) + y_hat1 * d
    p_hat = cross_val_predict(model_d, x, d, cv=cv, method="predict_proba", n_jobs=-1)[
        :, 1
    ]
    p_hat = np.clip(p_hat, trimming, 1 - trimming)

    # Step 3: build doubly robust score, then average for the ATE.
    dr_score = y_hat1 - y_hat0 + (y - y_hat_obs) * (d / p_hat - (1 - d) / (1 - p_hat))
    theta_hat = np.mean(dr_score)
    se_hat = np.sqrt(np.var(dr_score) / x.shape[0])

    return float(theta_hat), float(se_hat)


def summary_row(
    name: str, estimate: float, stderr: float, y_res: np.ndarray, d_res: np.ndarray
):
    return {
        "model": name,
        "estimate": estimate,
        "stderr": stderr,
        "lower_95": estimate - 1.96 * stderr,
        "upper_95": estimate + 1.96 * stderr,
        "rmse_y_res": np.sqrt(np.mean(y_res**2)),
        "rmse_d_res": np.sqrt(np.mean(d_res**2)),
    }


def main(degree_control=2) -> None:
    # ---------------------------------------------------------------------
    # 1) Load data and define treatment indicator used in this analysis.
    # ---------------------------------------------------------------------
    data = pd.read_csv(
        "https://raw.githubusercontent.com/VC2015/DMLonGitHub/master/penn_jae.dat",
        sep=r"\s+",
    )
    n, p = data.shape
    data = data[data["tg"].isin([0, 4])].copy()
    data = data[data["inuidur1"] > 0].copy()

    data["T4"] = np.where(data["tg"] == 4, 1, 0)

    print(f"Raw shape: n={n}, p={p}")
    print(f"Filtered shape (tg in {{0,4}}): {data.shape}")
    print("Treatment counts (T4):")
    print(data["T4"].value_counts().sort_index())

    # ---------------------------------------------------------------------
    # 2) Build outcome/treatment arrays and engineer a rich control set.
    # ---------------------------------------------------------------------
    # Work with log duration to reduce sensitivity to long right tails.
    y = np.log(data["inuidur1"].to_numpy())
    d = data["T4"].to_numpy()

    # Use second-order interactions to flexibly capture confounding structure.
    # This large set of controls will be fed into ML models to flexibly adjust for confounding.
    control_formula = (
        "0 + (female + black + othrace + C(dep) + q2 + q3 + q4 + q5 + q6 "
        f"+ agelt35 + agegt54 + durable + lusd + husd)**{degree_control}"
    )

    x_raw = dmatrix(control_formula, data=data, return_type="dataframe")
    # Remove near-linear dependencies before feeding controls into ML models.
    x = qr_decomposition(x_raw)

    print(f"Control matrix shape after QR de-collinearity filter: {x.shape}")
    print()

    # Baseline contrasts for context before orthogonalized estimates.
    naive = (
        data.loc[data["T4"] == 1, "inuidur1"].mean()
        - data.loc[data["T4"] == 0, "inuidur1"].mean()
    )
    naive_log = np.mean(y[d == 1]) - np.mean(y[d == 0])
    naive_log_pct = 100 * (np.exp(naive_log) - 1)

    print(f"Naive difference in means of inuidur1 (T4 - Control): {naive:.3f}")
    print(
        "Naive difference in means of log(inuidur1) "
        f"(T4 - Control): {naive_log:.3f} ({naive_log_pct:.2f}%)"
    )
    print()

    # ---------------------------------------------------------------------
    # 3) Define nuisance learners used in PLR and IRM.
    # ---------------------------------------------------------------------
    cv = KFold(n_splits=5, shuffle=True, random_state=123)
    # Feature engineering of data is done in 2) and FormulaTransformer is not needed here.
    # Some regressors (e.g. Lasso) require standardization for good performance.

    # Lasso/logit
    lasso_y = make_pipeline(StandardScaler(), LassoCV(cv=cv))
    logit_d = make_pipeline(
        StandardScaler(), LogisticRegressionCV(cv=cv, max_iter=5000)
    )

    # Decision tree
    dtr_y = make_pipeline(DecisionTreeRegressor(min_samples_leaf=10, ccp_alpha=0.001))
    dtr_d = make_pipeline(DecisionTreeClassifier(min_samples_leaf=10, ccp_alpha=0.001))

    # Random forest
    rf_y = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=10, random_state=123, n_jobs=-1
    )
    rf_d = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, random_state=123, n_jobs=-1
    )

    # Boosted trees
    gb_y = GradientBoostingRegressor(max_depth=2, random_state=123)
    gb_d = GradientBoostingClassifier(max_depth=2, random_state=123)

    # ---------------------------------------------------------------------
    # 4) Estimate PLR DML under several learner choices.
    # ---------------------------------------------------------------------
    plr_rows = []
    plr_specs = [
        ("lasso/logistic", lasso_y, logit_d, True),
        ("decision tree", dtr_y, dtr_d, True),
        ("random forest", rf_y, rf_d, True),
        ("boosted trees", gb_y, gb_d, True),
    ]

    for name, model_y, model_d, is_classifier in plr_specs:
        est, se, y_res, d_res = dml_plr(
            x,
            d,
            y,
            model_y,
            model_d,
            nfolds=5,
            classifier=is_classifier,
        )
        plr_rows.append(summary_row(name, est, se, y_res, d_res))

    plr_table = pd.DataFrame(plr_rows).set_index("model")
    print("PLR DML results (log outcome):")
    print(plr_table)
    print()

    best_y_model = plr_table["rmse_y_res"].idxmin()
    best_y_rmse = plr_table.loc[best_y_model, "rmse_y_res"]
    best_d_model = plr_table["rmse_d_res"].idxmin()
    best_d_rmse = plr_table.loc[best_d_model, "rmse_d_res"]

    print(
        "Best nuisance fit by RMSE:\n"
        f"Y predictor: {best_y_model} ({best_y_rmse:.4f}),\n"
        f"D predictor: {best_d_model} ({best_d_rmse:.4f})."
    )
    print()

    # ---------------------------------------------------------------------
    # 5) Estimate IRM DML under the same learner choices.
    # ---------------------------------------------------------------------
    irm_rows = []
    irm_specs = [
        ("lasso/logistic", lasso_y, lasso_y, logit_d),
        ("decision tree", dtr_y, dtr_y, dtr_d),
        ("random forest", rf_y, rf_y, rf_d),
        ("boosted trees", gb_y, gb_y, gb_d),
    ]

    for name, model_y0, model_y1, model_d in irm_specs:
        est, se = dml_irm(
            x,
            d,
            y,
            model_y0,
            model_y1,
            model_d,
            trimming=0.01,
            nfolds=5,
        )
        irm_rows.append(
            {
                "model": name,
                "estimate": est,
                "stderr": se,
                "lower_95": est - 1.96 * se,
                "upper_95": est + 1.96 * se,
            }
        )

    irm_table = pd.DataFrame(irm_rows).set_index("model")
    print("IRM DML results (log outcome):")
    print(irm_table)


if __name__ == "__main__":
    main(degree_control=2)
