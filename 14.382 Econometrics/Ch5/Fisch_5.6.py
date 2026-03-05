import polars as pl
import numpy as np
from statsmodels.sandbox.regression.gmm import GMM
import matplotlib.pyplot as plt

# Data sources: Saint Louis Fed and Yahoo Finance
# URL: https://www.stlouisfed.org/, http://finance.yahoo.com/

#   Description of the data: monthly data from 1959:M01 to 2015:M01 (675 obs) of the following variables:
# - PCEND: Personal Consumption Expenditures: Nondurable Goods (billions of dollars) from FED
# - PPCEND: Personal Consumption expenditures: Nondurable goods (chain-type price index), DNDGRG3M086SBEA from FED
# - CNP16OV: Civilian Noninstitutional Population (thousands of persons) from FED
# - GS1: annualized 1-Year Treasury Constant Maturity Rate from FED
# - SP500: S&P 500 index at closing price of the last day of the month, ^GSPC from Yahoo Finance

# Using the data provided, provide GMM-based estimation and specification
# testing for the consumption CAPM model along the lines of Section (4.5),
# with the difference being that you try
#
# (a) using only B(Z_t)= 1 as the technical instrument (this should give us m=d), and
# (b) only B(Z_t)= (1, R_1,t, R_2,t) as the technical instruments.
#
# You can begin by replicating Section 5.5 (we used gmm package in R; if you
# use different packages, the results may differ, but hopefully not by much),
# but please don’t report the replication results.
#
# Provide detailed explanations for what you are doing; the Hansen and
# Singleton’s article is a good example of how you could write things up.
####################


case = None  # case to run: 'a' for B(Z_t)=1, 'b' for B(Z_t)=(1, R_1,t, R_2,t), None for own lag and interactions
# If case = None, specify nlags and interactions below:
nlags = 1  # number of lags for instruments
interactions = False  # whether to include interaction terms in instruments
n_iter = (
    50  # number of iterations for iterated GMM (or 'cue' for continuously updated GMM)
)
# Part I: Processing Data

# Reading the data

data = (
    pl.read_csv(
        "ccapm-long_mert.csv",
        has_header=True,
        separator=",",
        eol_char="\r",
        try_parse_dates=True,
    )
    .with_columns(
        # Extract year, month, day from incorrect date
        pl.col("DATE").dt.year().alias("year_2d"),
        pl.col("DATE").dt.month().alias("month"),
        pl.col("DATE").dt.day().alias("day"),
    )
    .with_columns(
        # Apply century correction rule
        pl.when(pl.col("year_2d") < 25)
        .then(2000 + pl.col("year_2d"))
        .otherwise(1900 + pl.col("year_2d"))
        .alias("year_corrected")
    )
    .with_columns(
        # Rebuild proper date
        pl.date(pl.col("year_corrected"), pl.col("month"), pl.col("day")).alias("DATE")
    )
    .drop(["year_2d", "month", "day", "year_corrected"])
    .sort("DATE")
)


# Preparing data
df = (
    data.with_columns(
        (pl.col("PCEND") / (pl.col("PPCEND") * pl.col("CNP16OV"))).alias("rCpc"),
        (pl.col("PPCEND") / pl.col("PPCEND").shift(12) - 1).alias("inflation"),
    )
    .with_columns(
        ((1 + pl.col("GS1") / 100 - pl.col("inflation")) ** (1 / 12)).alias(
            "Rb"
        ),  # total monthly return to bonds (deflated)
        (
            pl.col("SP500")
            / pl.col("SP500").shift(1)
            * pl.col("PPCEND").shift(1)
            / pl.col("PPCEND")
        ).alias(
            "Rm"
        ),  # total monthly return to stocks (deflated)
        (pl.col("rCpc") / pl.col("rCpc").shift(1)).alias(
            "Rc"
        ),  # total monthly return to per-capita consumption (deflated)
    )
    .select(["Rc", "Rb", "Rm"])
    .drop_nulls()
)

# Control plot of prepared data
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(df[499:599, 0], color="black", label="Consumption")
ax.plot(df[499:599, 2], color="green", label="Bond")
ax.plot(df[499:599, 1], color="blue", label="Stock")
ax.set_title("consumption, stock, and bond total returns")
ax.legend()
ax.set_xlabel("Time")
plt.tight_layout()
# plt.savefig('L4/SeriesPlot.pdf')
plt.show()


# Create instruments based on the case
if case == "a":
    instruments = [pl.lit(1).alias("const")]  # constant only
elif case == "b":
    instruments = [pl.lit(1).alias("const")]  # constant
    for lag in range(1, nlags + 1):
        instruments.append(pl.col("Rc").shift(lag).alias(f"Rc_lag{lag}"))
        instruments.append(pl.col("Rb").shift(lag).alias(f"Rb_lag{lag}"))
        instruments.append(pl.col("Rm").shift(lag).alias(f"Rm_lag{lag}"))
else:
    # Create instruments  matrix with constant and lagged returns
    instruments = [pl.lit(1).alias("const")]  # constant

    for lag in range(1, nlags + 1):
        instruments.append(pl.col("Rc").shift(lag).alias(f"Rc_lag{lag}"))
        instruments.append(pl.col("Rb").shift(lag).alias(f"Rb_lag{lag}"))
        instruments.append(pl.col("Rm").shift(lag).alias(f"Rm_lag{lag}"))
        if interactions:
            instruments.append(
                (pl.col("Rc").shift(lag) * pl.col("Rm").shift(lag)).alias(
                    f"Rc_Rm_lag{lag}"
                )
            )
            instruments.append(
                (pl.col("Rc").shift(lag) * pl.col("Rb").shift(lag)).alias(
                    f"Rc_Rb_lag{lag}"
                )
            )
            instruments.append(
                (pl.col("Rm").shift(lag) * pl.col("Rb").shift(lag)).alias(
                    f"Rm_Rb_lag{lag}"
                )
            )
            instruments.append((pl.col("Rc").shift(lag) ** 2).alias(f"Rc_2_lag{lag}"))
            instruments.append((pl.col("Rm").shift(lag) ** 2).alias(f"Rm_2_lag{lag}"))
            instruments.append((pl.col("Rb").shift(lag) ** 2).alias(f"Rb_2_lag{lag}"))

# Combine returns and instruments
x = df.select(["Rc", "Rb", "Rm"] + instruments).drop_nulls()

endog = x.to_numpy()[:, :3]  # consumption, bond, market return growth (Rc, Rb, Rm)
exog = x.to_numpy()[:, 3:]  # instruments


class HansenSingletonGMM(GMM):
    def momcond(self, params):
        """
        Defines the moment conditions for the Euler equation.
        params: array-like containing [beta, alpha]
        """
        beta, alpha = params

        # Unpack endogenous variables (Rc, Rb, Rm)
        c_growth = self.endog[:, 0]
        bond_return = self.endog[:, 1]
        market_return = self.endog[:, 2]

        # Calculate the Euler equation pricing errors
        m1 = beta * c_growth ** (-alpha) * bond_return - 1
        m2 = beta * c_growth ** (-alpha) * market_return - 1

        # Multiply the error by the instruments (Z) to get the moment conditions.
        g1 = m1[:, None] * self.instrument  # element-wise multiplication
        g2 = m2[:, None] * self.instrument

        moments = np.hstack([g1, g2])

        return moments


# Initialize the model with endog, exog, and no initial instrument weighting matrix
model = HansenSingletonGMM(endog, exog=exog, instrument=exog, k_moms=exog.shape[1] * 2)

# Fit the model
# maxiter determines how many times the weighting matrix is updated (iterated GMM)
res = model.fit(
    start_params=[0.99, 1.0],
    maxiter=n_iter,
    optim_method="nm",
    inv_weights=None,
    weights_method="cov",  # covariance estimator
    wargs={},
)

# Print the results
print(res.summary(yname="Euler Eq", xname=["beta", "alpha"]))
print(res.jtest())
# Print LaTeX representation of the summary table
print(res.summary(yname="Euler Eq", xname=["beta", "alpha"]).as_latex())

# Run the J-test for overidentifying restrictions
j_stat, p_value, _ = res.jtest()

print("--- Hansen's J-Test ---")
print(f"J-statistic: {j_stat:.4f}")
print(f"p-value:     {p_value:.4f}")

# Additional diagnostics: show statsmodels' stored jval and the conventional Hansen J
try:
    stored_jval = getattr(res, "jval", None)
    nobs = getattr(res, "nobs", model.nobs)
    print("\nDiagnostics:")
    print("statsmodels stored jval (no n factor):", stored_jval)
    print(
        "Conventional Hansen J (n * stored jval):",
        None if stored_jval is None else nobs * stored_jval,
    )

    # manual calculation of J to control ddof/centering
    moms_at_est = model.momcond(res.params)
    gbar = moms_at_est.mean(axis=0)
    S_ddof0 = np.cov(moms_at_est, rowvar=False, ddof=0)
    S_ddof1 = np.cov(moms_at_est, rowvar=False, ddof=1)
    invS0 = np.linalg.pinv(S_ddof0)
    invS1 = np.linalg.pinv(S_ddof1)
    J_manual_ddof0 = moms_at_est.shape[0] * (gbar @ invS0 @ gbar)
    J_manual_ddof1 = moms_at_est.shape[0] * (gbar @ invS1 @ gbar)
    print("Manual J (ddof=0, divide by n):", J_manual_ddof0)
    print("Manual J (ddof=1, divide by n-1):", J_manual_ddof1)
    print("Moment count (nmoms):", moms_at_est.shape[1])
    print("nobs used for moments:", moms_at_est.shape[0])
except Exception as e:
    print("Diagnostics failed:", e)
