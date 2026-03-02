import polars as pl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

ROBUST_SE = True  # Set to True to use robust SEs
# Not sure which SE to use from the question. The qualitative conclusions do not change either way.

df_full = pl.read_csv("pset2_data.csv").with_columns(
        pl.col("OQB").eq(4).cast(pl.Int64).alias("Q4"),
        pl.col('logwage').ge(6).cast(pl.Int64).alias("Y_tilde")
    )
state_controls = [col for col in df_full.columns if col.startswith("SOB")]

df = df_full.select("logwage", "Q4", "Y_tilde")

print("Comments about questions in the code. Derivation as separate pdf.")

""" a) Sampling Distribution of OLS Estimator """
def run_regression(sample):
    if sample["Q4"].nunique() < 2:
        return None
    
    y = sample["logwage"]
    X = sm.add_constant(sample["Q4"])

    try:
        model = sm.OLS(y, X).fit()
        return model.params["Q4"]
    
    except:
        return None

# sample size
sample_sizes = [25, 100, 400, 1600]
S = 2500

results = {}

for n in sample_sizes:
    beta_no_replacement = []
    beta_with_replacement = []
    
    # without replacement
    i = 0
    while i < S:
        sample = df.sample(n, with_replacement=False, seed=i).to_pandas()
        beta = run_regression(sample)
        if beta is not None:
            beta_no_replacement.append(beta)
            i += 1

    # with replacement
    i = 0
    while i < S:
        sample = df.sample(n, with_replacement=True, seed=i).to_pandas()
        beta = run_regression(sample)
        if beta is not None:
            beta_with_replacement.append(beta)
            i += 1

    results[n] = {
        "without": np.array(beta_no_replacement),
        "with": np.array(beta_with_replacement)
    }

# Results plotted
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, n in enumerate(sample_sizes):
    ax = axes[i]
    ax.hist(results[n]["without"], bins=51, alpha=0.6, label="Without Replacement", density=True, color='C0')
    ax.hist(results[n]["with"], bins=51, alpha=0.6, label="With Replacement", density=True, color='C1')
    ax.axvline(results[n]["without"].mean(), linestyle='-', color='C0')
    ax.axvline(results[n]["with"].mean(), linestyle='-.', color='C1')
    ax.set_title(f"n = {n}")
    ax.set_xlabel(r"$\beta_{Q4}$")
    ax.set_ylabel("Density")
    ax.legend()

fig.suptitle("Distributions of Q4 Coefficient by Sample Size", fontsize=16)
plt.tight_layout()
plt.show()

# The sampling distributions look very similar between with and without replacement and their means are very close.
# Therefore, sampling with replacement is a good approximation for sampling without replacement.

""" d) t statistic """
def plug_in_estimator(sample):
    if sample['Q4'].n_unique() < 2:
        return None, None
    
    p1 = sample.filter(pl.col("Q4").eq(1)).select(pl.col("Y_tilde")).to_series().mean()
    p0 = sample.filter(pl.col("Q4").eq(0)).select(pl.col("Y_tilde")).to_series().mean()
    beta_hat = p1 - p0
    n1 = sample.filter(pl.col("Q4").eq(1)).select(pl.len()).item()
    n0 = sample.filter(pl.col("Q4").eq(0)).select(pl.len()).item()
    se_beta = np.sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)
    return beta_hat, se_beta

def bootstrap_t_statistic(df, beta_full, estimator, sample_sizes = [25, 50, 100, 200, 400], S=500):
    results = {}
    for n in sample_sizes:
        t_values = []
        i = 0
        while i < S:
            sample = df.sample(n, with_replacement=True, seed=i)
            beta_hat, se_beta = estimator(sample)
            if (beta_hat is None) or (se_beta == 0):
                continue
            t_stat = (beta_hat - beta_full) / se_beta
            t_values.append(t_stat)
            i += 1
        results[n] = np.array(t_values)
    return results

def plot_bootstrap_t_statistics(results, sample_sizes=[25, 50, 100, 200, 400], label=""):
    x = np.linspace(-4, 4, 400)
    std_pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    nrows = (len(sample_sizes) + 2 - 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharey=True)
    axes = axes.ravel()

    for i, n in enumerate(sample_sizes):
        ax = axes[i]
        tvals = results[n]
        ax.hist(tvals, bins=51, alpha=0.6, density=True, label="Bootstrapped t-statistic")
        ax.axvline(np.mean(tvals), linestyle='-', color='green', label="Mean")
        ax.plot(x, std_pdf, 'k-', label='Standard normal')
        ax.set_title(f"n = {n}")
        ax.set_xlabel("t-statistic")
        ax.set_ylabel("Density")
        ax.legend()

    # Hide any unused subplots
    for ax in axes[len(sample_sizes):]:
        ax.set_visible(False)
    fig.suptitle("t-Statistics by Sample Size" + label, fontsize=16)
    plt.tight_layout()
    plt.show()

# Full sample beta_0
beta_full, se_full = plug_in_estimator(df)
print(f"Full sample beta: {beta_full}, SE: {se_full}")

results = bootstrap_t_statistic(df, beta_full, plug_in_estimator)
plot_bootstrap_t_statistics(results, label = ": logwage >= 6")

# The distribution looks reasonably normal from n=100 onwards but further increases in sample size would certainly help.

""" e) log weekly wages """
def regression_e(sample):
    sample = sample.to_pandas()

    if sample["Q4"].nunique() < 2:
        return None, None
    
    y = sample["logwage"]
    X = sm.add_constant(sample["Q4"])

    try:
        if ROBUST_SE:
            model = sm.OLS(y, X).fit(cov_type='HC1')
        else:
            model = sm.OLS(y, X).fit()

        return model.params["Q4"], model.bse['Q4']
    
    except:
        return None, None

X_full = sm.add_constant(df.select("Q4").to_pandas()["Q4"])
y_full = df.select("logwage").to_pandas()["logwage"]
model_full = sm.OLS(y_full, X_full).fit()
beta_full = model_full.params["Q4"]

results = bootstrap_t_statistic(df, beta_full, regression_e)
plot_bootstrap_t_statistics(results, label = ": logwage")

# Interestingly, the distribution looks more noisy here, even at larger sample sizes.  
# This may be due to the fact that logwage is a continuous variable with potentially more variability than 
# the binary outcome in part d) while the coefficient is still on a binary birth quarter dummy.
# However, it has fewer outliers at small n and it becomes more normal looking at around the same n=100.
# But it could definitely benefit from even larger sample sizes.


""" f) controlling for birth state"""
# Version 1
def regression_f(sample):
    if sample["Q4"].n_unique() < 2:
        return None, None
    # Drop state controls that are all 0 
    sums = sample.select(pl.col(state_controls).sum()).to_dict()
    states = [c for c, v in sums.items() if v[0] > 0]
    # print(set(state_controls) - set(states))
    if len(states) == 0:
        return None, None
    sample = sample.select("logwage", "Q4", *states).to_pandas()    
    y = sample["logwage"]
    # I interpret 'rather than just include a constant...' as we should not include a constant
    # X = sm.add_constant(sample[["Q4"] + states])
    X = sample[["Q4"] + states]

    try:
        if ROBUST_SE:
            model = sm.OLS(y, X).fit(cov_type='HC1')
        else:
            model = sm.OLS(y, X).fit()
        return model.params["Q4"], model.bse['Q4']

    except np.linalg.LinAlgError:
        return None, None

# No state needs to be dropped in full sample
# X_full = sm.add_constant(df_full.to_pandas()[["Q4"] + state_controls])
X_full = df_full.to_pandas()[["Q4"] + state_controls]
y_full = df_full.select("logwage").to_pandas()["logwage"]
model_full = sm.OLS(y_full, X_full).fit()
beta_full = model_full.params["Q4"]

results = bootstrap_t_statistic(df_full, beta_full, regression_f)
plot_bootstrap_t_statistics(results, label = ": logwage + state controls")

# At small n the distribution is completely off with many outliers.
# This may be due to the increased complexity of the model with many state controls.
# With n=100 this has improved and looks kind of normal but more samples would be better as it 
# still looks worse than the previous parts.


""" g) conclusion """

# continuous vs binary: The continuous target variable in e) features less outliers at small sample sizes compared to the binary target variable in d). 
#                       However, as sample size increases, the advantage diminishes, maybe due to the higher variability in the continuous outcome.
#                       Both appear reasonably normal from n=100 onwards, but the continuous target is generally more noisy.
# including controls:   Including state controls increases the complexity of the model, which leads to huge outliers at small sample sizes and is much less normal compared to part e).
#                       Much larger sample sizes would be needed to achieve a normal distribution when including many controls.