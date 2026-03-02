import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
N = 50  # Number of agents
T = 10.0  # Total time
dt = 0.01  # Time step
steps = int(T / dt)
sigma = 0.5  # Volatility (Noise temperature)


def simulate_market(J, title):
    # Initialize agents randomly around 0
    X = np.random.normal(0, 0.1, N)

    # Store history for plotting
    history = np.zeros((steps, N))

    for t in range(steps):
        # 1. Fundamental Utility Force (Double-Well Potential)
        # Derivative of V(x) = -x^2/2 + x^4/4 is -x + x^3.
        # Force is -dV/dx = x - x^3
        F_fund = X - X**3

        # 2. Social Alignment Force (Mean Field)
        # Pulls agent i toward the average of all agents
        mean_X = np.mean(X)
        F_social = J * (mean_X - X)

        # 3. Stochastic Update (Euler-Maruyama)
        dW = np.random.normal(0, np.sqrt(dt), N)
        X = X + (F_fund + F_social) * dt + sigma * dW

        history[t, :] = X

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    time_axis = np.linspace(0, T, steps)

    # Plot individual trajectories
    for i in range(N):
        plt.plot(time_axis, history[:, i], lw=0.8, alpha=0.6)

    # Plot the Mean Field (Market Sentiment)
    plt.plot(
        time_axis,
        np.mean(history, axis=1),
        color="black",
        lw=3,
        linestyle="--",
        label="Mean Field (Aggregate)",
    )

    plt.title(f"{title} (Coupling J = {J})", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Agent State (Sentiment)")
    plt.ylim(-2.5, 2.5)
    plt.axhline(1, color="red", alpha=0.3, ls=":", label="Equilibrium A (Bull)")
    plt.axhline(-1, color="blue", alpha=0.3, ls=":", label="Equilibrium B (Bear)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.show()


# --- SCENARIO 1: Low Coupling (Rationality Dominates) ---
# Agents independently explore both wells. The mean hovers near 0.
simulate_market(J=0.1, title="Scenario 1: Disordered Phase (Idiosyncratic)")

# --- SCENARIO 2: High Coupling (Herding Dominates) ---
# Agents lock into one equilibrium together. Spontaneous Symmetry Breaking.
simulate_market(J=2.5, title="Scenario 2: Ordered Phase (Herding)")


""" More"""
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters for Guaranteed Cycling ---
T = 200.0
dt = 0.05
steps = int(T / dt)

# Timescale separation
epsilon = 0.08  # Small = Fast Jumps, Slow Recovery

# Model Coefficients (FitzHugh-Nagumo Topology)
# These ensure the fixed point is unstable
a = 0.7
b = 0.8
sigma = 0.15  # Noise level

# --- Initialization ---
m = np.zeros(steps)  # Market Sentiment
gamma = np.zeros(steps)  # Liquidity Constraints

# Start in a "Boom"
m[0] = 2.0
gamma[0] = 0.0

# --- Simulation ---
np.random.seed(42)

for t in range(steps - 1):
    # 1. FAST DYNAMICS (Cubic Nullcline - S-Shape)
    # dm = [ m - m^3/3 - gamma ] / epsilon
    # Econ Logic: Momentum vs. Fundamental Correction vs. Liquidity Drag

    drift_m = (m[t] - (m[t] ** 3) / 3 - gamma[t]) / epsilon
    diffusion_m = sigma / np.sqrt(epsilon)

    m[t + 1] = m[t] + drift_m * dt + diffusion_m * np.random.normal(0, np.sqrt(dt))

    # 2. SLOW DYNAMICS (Linear Recovery)
    # dgamma = m + a - b*gamma
    # Econ Logic: High activity (m) tightens liquidity (gamma rises).
    # Recession (low m) restores liquidity (gamma falls).

    d_gamma = m[t] + a - b * gamma[t]
    gamma[t + 1] = gamma[t] + d_gamma * dt

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: The Boom-Bust Time Series
time = np.linspace(0, T, steps)
ax1.plot(time, m, color="navy", lw=1.5, label="Sentiment ($m$)")
ax1.plot(
    time,
    gamma,
    color="crimson",
    ls="--",
    lw=1.5,
    label="Liquidity Constraint ($\gamma$)",
)
ax1.set_title("Endogenous Cycles (Boom - Crash - Recovery)")
ax1.set_xlabel("Time")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

# Plot 2: The Limit Cycle (Hysteresis)
ax2.plot(m, gamma, color="black", lw=1.5)
ax2.set_title("The Hysteresis Loop")
ax2.set_xlabel("Sentiment ($m$)")
ax2.set_ylabel("Liquidity Constraint ($\gamma$)")
ax2.grid(True, alpha=0.3)

# Add the "Fast Nullcline" (The S-Curve) for visual intuition
m_range = np.linspace(-2.5, 2.5, 100)
nullcline = m_range - (m_range**3) / 3
ax2.plot(
    m_range,
    nullcline,
    color="green",
    alpha=0.3,
    ls=":",
    label="Theoretical Stability Curve",
)
ax2.legend()

plt.tight_layout()
plt.show()


""""""
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
T = 1600.0
dt = 0.05
steps = int(T / dt)

# Timescale separation
epsilon = 0.2

# FitzHugh-Nagumo Parameters (Tuned for Excitability)
# "a" determines if we cycle or wait.
# a = 0.7 -> Limit Cycle (Periodic)
# a = 1.05 -> Excitable (Stable Point near the cliff)
a = 1.05
b = 0.5
sigma = 0.25  # Noise must be strong enough to kick it over the barrier

# --- Initialization ---
m = np.zeros(steps)  # Sentiment
gamma = np.zeros(steps)  # Liquidity
m[0] = -1.0  # Start in the "Depressed" stable state
gamma[0] = -0.5

# --- Simulation ---
np.random.seed(42)

for t in range(steps - 1):
    # 1. Fast Dynamics (S-Curve)
    drift_m = (m[t] - (m[t] ** 3) / 3 - gamma[t]) / epsilon
    diffusion_m = sigma / np.sqrt(epsilon)

    # 2. Slow Dynamics
    # This keeps us stable unless m gets kicked high enough
    d_gamma = m[t] + a - b * gamma[t]

    # Update
    m[t + 1] = m[t] + drift_m * dt + diffusion_m * np.random.normal(0, np.sqrt(dt))
    gamma[t + 1] = gamma[t] + d_gamma * dt

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Time Series
time = np.linspace(0, T, steps)
ax1.plot(time, m, color="navy", lw=1)
ax1.set_title("Poisson Crashes (Random Arrival)")
ax1.set_ylabel("Market Sentiment")
ax1.set_xlabel("Time")
ax1.grid(True, alpha=0.3)
# Mark the "Rest" state
ax1.axhline(-1, color="green", ls="--", alpha=0.5, label="Metastable Equilibrium")
ax1.legend()

# Plot 2: Phase Plane
ax2.plot(m, gamma, color="black", lw=0.8, alpha=0.6)
ax2.set_title("Excursions in Phase Space")
ax2.set_xlabel("Sentiment ($m$)")
ax2.set_ylabel("Liquidity ($\gamma$)")
ax2.grid(True, alpha=0.3)

# Draw Nullclines to show why it's excitable
x_range = np.linspace(-3, 3, 100)
nullcline_fast = x_range - (x_range**3) / 3
nullcline_slow = (x_range + a) / b
ax2.plot(x_range, nullcline_fast, "g--", alpha=0.5, label="Nullcline m")
ax2.plot(x_range, nullcline_slow, "r--", alpha=0.5, label="Nullcline gamma")
ax2.legend()

plt.tight_layout()
plt.show()
