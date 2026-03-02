import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pl.read_csv("ahs_cleaned_for_prediction_rashomonready.csv")

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare features and target
X_train = train_df.drop("LOGVALUE")
y_train = train_df["LOGVALUE"]
X_test = test_df.drop("LOGVALUE")
y_test = test_df["LOGVALUE"]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled data back to DataFrame
X_train_scaled = pl.DataFrame(X_train_scaled, schema=X_train.schema)
X_test_scaled = pl.DataFrame(X_test_scaled, schema=X_train.schema)

# Combine target and features for training
train_data = pl.concat([y_train.to_frame(), X_train_scaled], how="horizontal")

# Fit Lasso regression with different alpha values and sample sizes
sizes = np.linspace(1, 0.25, 4)  # Sample sizes
alphas = np.logspace(-2.8, -0.5, 10)  # Regularization strengths
coefficients = {}
scores = {}

for s in sizes:
    train_sample = train_data.sample(fraction=s, seed=42)  # Sample data

    X_train_sampled = train_sample.drop("LOGVALUE")
    y_train_sampled = train_sample["LOGVALUE"]

    for a in alphas:
        lasso = Lasso(alpha=a, random_state=42)  # Initialize Lasso model
        lasso.fit(X_train_sampled, y_train_sampled)  # Fit model

        coefficients[(s, a)] = lasso.coef_  # Store coefficients

        train_score = lasso.score(X_train_sampled, y_train_sampled)  # Train score
        test_score = lasso.score(X_test_scaled, y_test)  # Test score

        scores[(s, a)] = (train_score, test_score)  # Store scores

# Create subplots for each sample size
fig, axes = plt.subplots(len(sizes), 1, figsize=(15, 2.5 * len(sizes)), sharex=True)
colors = plt.cm.winter(np.linspace(0, 1, len(alphas)))
if len(sizes) == 1:
    axes = [axes]

for idx, s in enumerate(sizes):
    ax = axes[idx]
    size_coefs = {a: coefficients[(s, a)] for a in alphas}

    coef_matrix = np.array(list(size_coefs.values()))
    x_pos = np.arange(len(X_train.columns))
    width = 0.8 / len(alphas)

    for alpha_idx, a in enumerate(alphas):
        offset = (alpha_idx - len(alphas) / 2) * width
        ax.bar(
            x_pos + offset,
            coef_matrix[alpha_idx, :],
            width,
            color=colors[alpha_idx],
        )
    ax.set_xlabel("Features")
    ax.set_ylabel("Coefficient Value")
    ax.set_title(f"Coefficients for Sample Size: {s:.2f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(X_train.columns, rotation=90, fontsize=6)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

# Create a single legend at the bottom
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(alphas))]
labels = [f"Alpha: {a:.2e}" for a in alphas]
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)

plt.tight_layout(rect=[0, 0.04, 1, 1])  # Adjust layout
plt.show()

# Prepare scores data for visualization
scores_data = []
for (s, a), (train_score, test_score) in scores.items():
    scores_data.append(
        {
            "sample_size": s,
            "alpha": a,
            "train_score": train_score,
            "test_score": test_score,
        }
    )
scores_df = pl.DataFrame(scores_data)

# Plot scores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for s in sizes:
    subset = scores_df.filter(pl.col("sample_size") == s)
    plt.plot(
        subset["alpha"],
        subset["train_score"],
        marker="o",
        label=f"Sample size: {s:.2f}",
    )
plt.xlabel("Alpha")
plt.ylabel("Train R²")
plt.xscale("log")
plt.legend()
plt.title("Train Scores")

plt.subplot(1, 2, 2)
for s in sizes:
    subset = scores_df.filter(pl.col("sample_size") == s)
    plt.plot(
        subset["alpha"], subset["test_score"], marker="o", label=f"Sample size: {s:.2f}"
    )
plt.xlabel("Alpha")
plt.ylabel("Test R²")
plt.xscale("log")
plt.legend()
plt.title("Test Scores")
plt.tight_layout()
plt.show()


"""
Rashomon Effect in Predictive Modelling

In predictive modeling, the Rashomon effect occurs when there are many different models that 
describe the same data similarly well (i.e., they have very similar predictive accuracy). 
Because the models use different features or weights to achieve the same result, they tell 
"different stories" about how the underlying system works.

The plot displays Lasso regression coefficients across different subsets of data (sample sizes) 
and different regularization penalties (represented by the colors).

- Regularization Strength (alpha):
Lasso regression applies an L1 penalty that forces with increasing alpha the coefficients of 
less important features to exactly zero (leading to a sparser model). Looking at the grouped 
bars for any specific feature on the x-axis, we can see how the coefficient for that feature 
changes with alpha. An increase in alpha might push the coefficient to zero, effectively 
dropping it from the model entirely. However, multiple values of alpha achieve a similar 
performance as measured by R² in the second plot.
This means you could have one model with 50 features and another model with only 30 features 
(higher alpha) that both predict housing prices with nearly identical accuracy. They are
equally "good" models, but they disagree entirely on which features actually matter.

- Sample Size:
We have four models trained on 100%, 75%, 50%, and 25% of the training data. Comparing the 
exact height of the bars (strength coefficients) across the four subplots, it is noticable 
that while the massive spikes stay relatively consistent, many of the smaller coefficients 
shift dramatically depending on the sample size. By simply changing the data sample slightly, 
the algorithm finds a new combination of features to optimize the prediction. In particular, 
if there are highly correlated features in the AHS dataset, the Lasso regression may almost
arbitrarily select one feature over another, but change its choice in a slightly different 
sample. Both models perform well, but the interpretation of what drives housing prices changes.
"""
