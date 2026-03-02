import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)
n_samples = 10000

# Generate latent confounder: poor lifestyle habits (V)
# Higher V means worse lifestyle habits
V = np.random.normal(0, 1, n_samples)

# Define True Potential Outcomes (Y)
# Note, smoking marijuana (D) is not in this equation, so the true effect is 0.
# Longevity = 80 years - impact of poor lifestyle habits + noise
epsilon = np.random.normal(0, 1, n_samples)
Y = 80 - 5 * V + epsilon

# Randomized Controlled Trial
# Smoking is assigned by a coin flip, totally independent of V
D_rct = np.random.binomial(1, 0.5, n_samples)

# Observational Study
# Probability of smoking increases with poor lifestyle habits (V)
# Use a sigmoid function to turn V into a probability
prob_smoking = 1 / (1 + np.exp(-2 * V))
D_obs = np.random.binomial(1, prob_smoking)

# Analyze Results
df = pd.DataFrame(
    {"Longevity": Y, "Smoking_RCT": D_rct, "Smoking_Obs": D_obs, "Health_Choices": V}
)

rct_effect = (
    df[df["Smoking_RCT"] == 1]["Longevity"].mean()
    - df[df["Smoking_RCT"] == 0]["Longevity"].mean()
)
obs_effect = (
    df[df["Smoking_Obs"] == 1]["Longevity"].mean()
    - df[df["Smoking_Obs"] == 0]["Longevity"].mean()
)

print(f"Results:")
print(f"True causal effect δ: 0.00")
print(f"Estimated predictive effect π (RCT): {rct_effect:.3f}")
print(f"Estimated predictive effect π (Observational): {obs_effect:.3f}")

# Visualization
# Scatter plot of effect of smoking on longevity in RCT vs Observational study
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Example: Impact of Smoking Marijuana on Longevity (Confounder: Poor Health Choices)",
    fontsize=16,
)
sns.scatterplot(
    data=df,
    x="Health_Choices",
    y="Longevity",
    hue="Smoking_RCT",
    palette="Set1",
    alpha=0.7,
    ax=axes[0],
)
axes[0].set_title("RCT")
axes[0].set_xlabel("Poor Health Choices (V)")
axes[0].set_ylabel("Longevity (Y)")
axes[0].legend(
    title="Smoking Status",
    labels=["Non-Smoker", "Smoker"],
    handles=axes[0].get_legend_handles_labels()[0],
)

sns.scatterplot(
    data=df,
    x="Health_Choices",
    y="Longevity",
    hue="Smoking_Obs",
    palette="Set1",
    alpha=0.7,
    ax=axes[1],
)
axes[1].set_title("Observational Study")
axes[1].set_xlabel("Poor Health Choices (V)")
axes[1].set_ylabel("Longevity (Y)")
axes[1].legend(
    title="Smoking Status",
    labels=["Non-Smoker", "Smoker"],
    handles=axes[1].get_legend_handles_labels()[0],
)

plt.tight_layout()
plt.show()

print(
    """This scatter plot visualizes the raw data from the observational study simulation to explain how selection bias works:
Confounder (V): The x-axis represents "poor health choices." You can see a clear negative trend—as health choices get worse (move right), longevity (Y) decreases.
Selection bias in observational design: the colors represent the smoking status. Note how the Smokers (blue) are concentrated on the right side of the graph (high V),
while Non-Smokers (red) are concentrated on the left (low V). Because smokers are also the ones with poor health choices, they appear at the bottom-right of the plot
(lower longevity). If you only look at the colors and longevity without accounting for V, you would wrongly conclude that smoking causes a decrease in life expectancy,
even though in our simulation, the true effect of smoking is zero.
In contrast, in the RCT plot, the smokers and non-smokers are evenly distributed across all levels of V, which is why the estimated effect is close to the true effect of zero."""
)

# Estimated effect
plt.figure(figsize=(5, 5))
plt.bar(
    ["RCT (Random)", "Observational (Self-Selected)"],
    [rct_effect, obs_effect],
    color=["green", "red"],
)
plt.axhline(0, color="black", lw=1)
plt.ylabel("Estimated Effect on Longevity (Years)")
plt.title("Selection Bias: RCT vs. Observational Estimates")
plt.show()

print(
    """This bar chart compares the estimated effect of smoking marijuana on longevity from the RCT and the observational study:
The RCT bar (green) is close to zero, which is the true causal effect in our simulation. This is because the RCT design successfully eliminates confounding by randomly
assigning smoking status, ensuring that smokers and non-smokers have similar distributions of the confounder (V).
The Observational bar (red) shows a large negative effect, which is a result of selection bias."""
)
