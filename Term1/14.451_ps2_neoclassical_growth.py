import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq

## Neoclassical Growth Model via Value Function Iteration
# Description: https://www.karenkopecky.net/Teaching/eco613614/Notes_ValueFunctionIteration.pdf

# Model Parameters
alpha = 1 / 3
beta = 0.95
delta = 0.15
epsilon = 1.5

# Maximum capital
k_bar = 1 / delta ** (1 / (1 - alpha))

# Simulation Parameters
tol = 5e-6
max_iter = 1000

# Capital grid: logspace since more non-linear for lower k
k_grid = np.logspace(np.log10(0.01), np.log10(k_bar), 2500)
# k_grid = np.linspace((0.01), (k_bar), 2500)

# Initial guess for value function
v0 = np.ones_like(k_grid)


# Econ functions
def U(c, epsilon=1):
    """Constant Relative Risk Aversion (CRRA) Utility Function"""
    u = np.full_like(c, -np.inf, dtype=float)  # default to -inf
    feasible = c > 0

    if epsilon != 1:
        u[feasible] = (
            ((c[feasible]) ** ((epsilon - 1) / epsilon) - 1) * epsilon / (epsilon - 1)
        )
    else:
        u[feasible] = np.log(c[feasible])

    return u


def f(k, alpha=alpha):
    """Cobb-Douglas Production Function"""
    return k**alpha


def V_step(v_grid, k_grid, beta=beta, delta=delta, epsilon=epsilon):
    """Value Function Iteration Step"""
    policy = []
    v_new = []
    for ki in k_grid:
        c = f(ki) + (1 - delta) * ki - k_grid
        T = U(c, epsilon) + beta * v_grid

        V_max = np.max(T)
        k_max = k_grid[np.argmax(T)]

        policy.append(k_max)
        v_new.append(V_max)

    return np.array(v_new), np.array(policy)


def convergence(v0, v1, tol=tol):
    d = np.max(np.abs(v1 - v0))
    return d < tol * np.linalg.norm(v1)


# Initialize
converged = False
it = 0
v_iter = [v0]
v = v0

while not converged and it < max_iter:
    v_next, policy = V_step(v, k_grid, beta=beta, delta=delta, epsilon=epsilon)
    v_iter.append(v_next)
    converged = convergence(v, v_next)
    v = v_next
    it += 1
print(f"Converged: {converged} in {it} iterations")


# Plot value function
plt.plot(k_grid, v, label="Value function")
plt.xlabel(r"Capital ($k$)")
plt.ylabel(r"Value Function ($V(k)$)")
plt.title("Value Function")
plt.legend()
plt.show()

# Find intersection point between policy function and 45-degree line using interpolation
interp_func = interp1d(k_grid, policy - k_grid)
k_intersect = brentq(interp_func, k_grid[0], k_grid[-1])
policy_intersect = interp1d(k_grid, policy)(
    k_intersect
)  # should be equal to k_intersect as on 45-degree line
print(f"Intersection at k = {k_intersect:.4f}, policy = {policy_intersect:.4f}")

# Analytical steady state (depends on f's inverse)
k_steady = ((1 / beta - 1 + delta) / alpha) ** (1 / (alpha - 1))
print(f"Analytical steady state at k = {k_steady:.4f}")

# Plot policy function
plt.plot(k_grid, policy, label="Policy function")
plt.plot(k_grid, k_grid, label="45-degree line", linestyle="--")
plt.scatter(
    k_intersect, policy_intersect, color="red", s=25, zorder=5, label="Steady State"
)
plt.xlabel(r"Current Capital ($k$)")
plt.ylabel(r"Next Period Capital ($k'$)")
plt.title("Optimal Policy Function")
plt.legend()
plt.show()

# Simulate trajectory starting from k = k_steady / 2
k0_below = k_steady / 2
k0_above = k_steady * 1.5

n_periods = 25
k_path_below = [k0_below]
k_path_above = [k0_above]

# Interpolate the policy function for arbitrary k
policy_func = interp1d(k_grid, policy, kind="linear", fill_value="extrapolate")

for t in range(n_periods):
    k_next_below = policy_func(k_path_below[-1])
    k_path_below.append(k_next_below)
    k_next_above = policy_func(k_path_above[-1])
    k_path_above.append(k_next_above)

plt.plot(k_path_below, marker="o", color="C0", label="Capital Path Below Steady State")
plt.plot(k_path_above, marker="o", color="C0")
plt.axhline(k_steady, color="gray", linestyle="--", label="Steady State")
plt.xlabel("Period")
plt.ylabel(r"Capital ($k$)")
plt.title(r"Exemplary Capital Trajectories")
plt.legend()
plt.show()


# Compare results when doubling epsilon
epsilon_double = epsilon * 2

# Re-run value function iteration with doubled epsilon
v0_double = np.ones_like(k_grid)
converged = False
it = 0
v_double = v0_double

while not converged and it < max_iter:
    v_next_double, policy_double = V_step(
        v_double, k_grid, beta=beta, delta=delta, epsilon=epsilon_double
    )
    converged = convergence(v_double, v_next_double)
    v_double = v_next_double
    it += 1

# Find intersection point for new policy
interp_func_double = interp1d(k_grid, policy_double - k_grid)
k_intersect_double = brentq(interp_func_double, k_grid[0], k_grid[-1])
policy_intersect_double = interp1d(k_grid, policy_double)(k_intersect_double)

# Analytical steady state for doubled epsilon (unchanged, as epsilon only affects utility curvature)
print(
    f"Doubled epsilon: Intersection at k = {k_intersect_double:.4f}, policy = {policy_intersect_double:.4f}"
)

# Plot comparison of policy functions
plt.plot(k_grid, policy, label=rf"Policy ($\varepsilon$={epsilon})")
plt.plot(k_grid, policy_double, label=rf"Policy ($\varepsilon$={epsilon_double})")
plt.plot(k_grid, k_grid, label="45-degree line", linestyle="--", color="gray")
plt.scatter(
    k_intersect,
    policy_intersect,
    color="red",
    s=35,
    zorder=5,
    label=rf"Steady State ($\varepsilon$={epsilon})",
)
plt.scatter(
    k_intersect_double,
    policy_intersect_double,
    color="green",
    s=20,
    zorder=6,
    label=rf"Steady State ($\varepsilon$={epsilon_double})",
)
plt.xlabel(r"Current Capital ($k$)")
plt.ylabel(r"Next Period Capital ($k'$)")
plt.title("Policy Function Comparison")
plt.legend()
plt.show()

# Plot comparison of value functions
plt.plot(k_grid, v, label=rf"Value Function ($\varepsilon$={epsilon})")
plt.plot(
    k_grid, v_double, label=rf"Value Function ($\varepsilon$={epsilon_double})", ls="--"
)
plt.xlabel(r"Capital ($k$)")
plt.ylabel(r"Value Function ($V(k)$)")
plt.title("Value Function Comparison")
plt.legend()
plt.show()
