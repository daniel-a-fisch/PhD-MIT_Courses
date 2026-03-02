import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
from scipy.integrate import quad

def bisection(a, b, f, tol=1e-6, max_iter=500, verbose=False, return_zero = False):
    if f(a) * f(b) > 0:
        if return_zero:
            return 0.
        else:
            raise ValueError("No sign change found in the interval [a, b]. Try different values.")

    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)

        if abs(fc) < tol or (b - a) / 2 < tol:
            if verbose:
                print(f"Converged in {i+1} iterations.")
            return c

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    raise ValueError("Did not converge within the maximum number of iterations.")

def solve_foc(lower_bound, mid_point, upper_bound, func, negative = True):
    """ Solve the FOC in two intervals and compare the resulting profits to decide which is maximizing."""
    p1 = bisection(lower_bound, mid_point, func, return_zero=True)
    p2 = bisection(mid_point, upper_bound, func, return_zero=True)
    profit1 = (p1 - c) * F(delta - a * p1) if negative else (p1 - c) * (1 - F(a * p1 - delta))
    profit2 = (p2 - c) * F(delta - a * p2) if negative else (p2 - c) * (1 - F(a * p2 - delta))
    if profit1 > profit2:
        return p1
    else:
        return p2

gamma = gumbel_r.stats(moments='m') # Euler-Mascheroni constant to de-mean the distribution

def F(x):
    return gumbel_r.cdf(x, loc=-gamma)

""" a) Plot the PDF of the type 1 extreme value distribution (Gumbel)"""
epsilon = np.linspace(-10, 10, 500)

# Standard type 1 extreme value distribution (Gumbel)
pdf_gumbel = gumbel_r.pdf(epsilon, loc=-gamma)

# Negative of the type 1 extreme value distribution: flip the sign of ε
pdf_negative_gumbel = gumbel_r.pdf(-epsilon, loc=-gamma)

# Plotting
plt.figure(figsize=(9, 6))
plt.plot(epsilon, pdf_gumbel, label='Standard Logit', color='blue', lw=2, linestyle='--')
plt.plot(epsilon, pdf_negative_gumbel, label='Negative Logit', color='red', lw=2, linestyle='-')

# Formatting
# plt.title("PDF of Type 1 Extreme Value Distribution: Standard vs. Negative Logit")
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$f(\epsilon)$')
plt.axvline(0, color='gray', linestyle=':', lw=1)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig("../../Apps/Overleaf/14.271_PS_3/figures/logit.png")
plt.show()

# This figure compares the standard logit model's error distribution (type I extreme value, or Gumbel) 
# to its negative counterpart. 
# The standard distribution is right-skewed, suggesting that high utility values are less likely, 
# while most consumer valuations are near or below the mode. 
# In contrast, the negative logit distribution is left-skewed, implying that low utility values are rarer, 
# and high utility outcomes (e.g., strong preferences) are more likely.


""" b) FOC for monopoly pricing under standard and negative logit models"""

# Parameters
delta = 4.5  # intrinsic utility of the product
# p^m increases in delta
# for small values of delta there might not be a solution
a = 1    # price sensitivity
# price elasticity increases in a
c = 5.5     # marginal cost

p_vals = np.linspace(0, 10, 200)  # range of prices

# LHS: Purchase probabilities
# Standard logit model
purchase_prob_logit = 1 - gumbel_r.cdf(a * p_vals - delta, loc=-gamma)
# Negative logit model
purchase_prob_neglogit = gumbel_r.cdf(delta - a * p_vals, loc=-gamma)

# RHS: Marginal cost
marg_cost_logit = a * (p_vals - c) * gumbel_r.pdf(a * p_vals - delta, loc=-gamma)
marg_cost_neglogit = a * (p_vals - c) * gumbel_r.pdf(delta - a * p_vals, loc=-gamma)

# Plotting
plt.figure(figsize=(9, 6))
plt.plot(p_vals, purchase_prob_logit, label='Std.: Benefit', color='blue', lw=2, linestyle='-')
plt.plot(p_vals, marg_cost_logit, label='Std.: Cost', color='blue', lw=2, linestyle='--')

plt.plot(p_vals, purchase_prob_neglogit, label='Neg.: Benefit', color='red', lw=2, linestyle='-')
plt.plot(p_vals, marg_cost_neglogit, label='Neg.: Cost', color='red', lw=2, linestyle='--')    

# plt.vlines(p_neg, 0, 1, color='gray', linestyle=':', lw=1)
# plt.vlines(p_std, 0, 1, color='gray', linestyle=':', lw=1)

# plt.title("FOC for Monopoly Pricing: Logit vs. Negative Logit")
plt.xlabel(r'$p^m$')
plt.ylabel('marginal value')
plt.axvline(0, color='gray', linestyle=':', lw=1)
plt.grid(False)
plt.legend()
plt.tight_layout()
#plt.savefig("../../Apps/Overleaf/14.271_PS_3/figures/monopoly_pricing.png")
plt.show()

""" c) Influence of delta and c on monopoly price, demand, consumer surplus, and deadweight loss"""

# Define the FOC
def foc_neg(p):
    return F(delta - a * p) - a * (p - c) * gumbel_r.pdf(delta - a * p, loc=-gamma)
def foc(p):
    return 1 - F(a * p - delta) - a * (p - c) * gumbel_r.pdf(a * p - delta, loc=-gamma)

# Solve FOC numerically
p_neg = bisection(0,10,foc_neg)
solve_foc(0, 5, 10, foc_neg, negative = True)
p_std = bisection(0,10,foc)
solve_foc(0, 5, 10, foc, negative = False)

"""Analysis for varying delta and c"""

delta_list = np.linspace(1,10,19)
c_list = np.linspace(0,6,25)

C, Delta = np.meshgrid(c_list, delta_list)

# Ensure costs are lower than delta
mask = Delta > C-2

# Initialize result arrays
P_neg = np.full_like(C, np.nan, dtype=float)
P_std = np.full_like(C, np.nan, dtype=float)

demand_neg = np.full_like(C, np.nan, dtype=float)
demand_std = np.full_like(C, np.nan, dtype=float)

consumer_surplus_neg = np.full_like(C, np.nan, dtype=float)
consumer_surplus_std = np.full_like(C, np.nan, dtype=float)

DWL_neg = np.full_like(C, np.nan, dtype=float)
DWL_std = np.full_like(C, np.nan, dtype=float)

# Compute results over the grid of params
for i in range(Delta.shape[0]):
    for j in range(Delta.shape[1]):
        if mask[i, j]:
            delta = Delta[i, j]
            c = C[i, j]

            P_neg[i, j] = solve_foc(c, delta/a *2, delta/a *3, foc_neg, negative=True)
            P_std[i, j] = solve_foc(c, delta/a *2, delta/a *3, foc, negative=False)

            demand_neg[i, j] = F(delta - a * P_neg[i, j])
            demand_std[i, j] = 1 - F(a * P_std[i, j] - delta)

            consumer_surplus_neg[i, j], error = quad(lambda p: F(delta-a*p), P_neg[i, j], 10)
            consumer_surplus_std[i, j], error = quad(lambda p: 1 - F(a*p - delta), P_std[i, j], 10)

            DWL_neg[i, j], _ = quad(lambda p: F(delta-a*p), c, P_neg[i, j])
            DWL_std[i, j], _ = quad(lambda p: 1-F(a*p-delta), c, P_std[i, j])

# --- Plot 3D ---
# Plot negative logit monopoly price
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(Delta, C, P_neg, cmap='Reds', label='Neg. Logit')
surf2 = ax1.plot_surface(Delta, C, P_std, cmap='Blues', label='Std. Logit')
ax1.set_xlabel(r'$\delta$')
ax1.set_ylabel('c')
ax1.set_zlabel(r'$p^m$')
# ax1.set_title(r'Parameter dependency of $p^m$')

# ax2 = fig.add_subplot(122, projection='3d')
# surf2 = ax2.plot_surface(Delta, C, P_std, cmap='Blues')
# ax2.set_xlabel(r'$\delta$')
# ax2.set_ylabel('c')
# ax2.set_zlabel(r'$p^m$')
# ax2.set_title(r'$p^m$ for Standard Logit')

plt.tight_layout()
plt.savefig("../../Apps/Overleaf/14.271_PS_3/figures/mon_comparison_3d.png")
plt.show()

# --- Plot 2D slices ---
# Select indices
row = 7 #Delta.shape[0] // 2  # Fix delta (horizontal slice)
col = 4 #C.shape[1] // 2      # Fix c (vertical slice)

c_vals = C[row, :]
delta_vals = Delta[:, col]

# --- Plot horizontal slice (delta fixed) ---
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
print(Delta[row, 0])
# Left: Monopoly price and demand
axs[0].plot(c_vals, P_neg[row, :], label=r'$p^m_{neg}$', color='red')
axs[0].plot(c_vals, P_std[row, :], label=r'$p^m_{std}$', color='blue')
axs[0].plot(c_vals, demand_neg[row, :], label=r'$q_{neg}$', color='red', ls=':')
axs[0].plot(c_vals, demand_std[row, :], label=r'$q_{std}$', color='blue', ls=':')
axs[0].set_xlabel('c')
axs[0].set_ylabel('Price / Demand')
# axs[0].set_title(rf'Monopoly Price and Demand ($\delta$ = {Delta[row, 0]:.1f})')
axs[0].legend()
axs[0].grid(True)

# Right: Consumer surplus and deadweight loss
axs[1].plot(c_vals, consumer_surplus_neg[row, :], label=r'$CS_{neg}$', color='red', ls ='-')
axs[1].plot(c_vals, consumer_surplus_std[row, :], label=r'$CS_{std}$', color='blue', ls='-')
axs[1].plot(c_vals, DWL_neg[row, :], label=r'$DWL_{neg}$', color='red', ls=':')
axs[1].plot(c_vals, DWL_std[row, :], label=r'$DWL_{std}$', color='blue', ls=':')
axs[1].set_xlabel('c')
axs[1].set_ylabel('CS / DWL')
# axs[1].set_title(rf'Consumer Surplus and DWL ($\delta$ = {Delta[row, 0]:.1f})')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("../../Apps/Overleaf/14.271_PS_3/figures/mon_comparison_slice_delta.png")

# --- Plot vertical slice (c fixed) ---
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
print(C[0, col])
# Left: Monopoly price and demand
axs[0].plot(delta_vals, P_neg[:, col], label=r'$p^m_{neg}$', color='red')
axs[0].plot(delta_vals, P_std[:, col], label=r'$p^m_{std}$', color='blue')
axs[0].plot(delta_vals, demand_neg[:, col], label=r'$q_{neg}$', color='red', ls=':')
axs[0].plot(delta_vals, demand_std[:, col], label=r'$q_{std}$', color='blue', ls=':')
axs[0].set_xlabel(r'$\delta$')
axs[0].set_ylabel('Price / Demand')
# axs[0].set_title(f'Monopoly Price and Demand (c = {C[0, col]:.1f})')
axs[0].legend()
axs[0].grid(True)

# Right: Consumer surplus and deadweight loss
axs[1].plot(delta_vals, consumer_surplus_neg[:, col], label=r'$CS_{neg}$', color='red', ls='-')
axs[1].plot(delta_vals, consumer_surplus_std[:, col], label=r'$CS_{std}$', color='blue', ls='-')
axs[1].plot(delta_vals, DWL_neg[:, col], label=r'$DWL_{neg}$', color='red', ls=':')
axs[1].plot(delta_vals, DWL_std[:, col], label=r'$DWL_{std}$', color='blue', ls=':')
axs[1].set_xlabel(r'$\delta$')
axs[1].set_ylabel('CS / DWL')
# axs[1].set_title(f'Consumer Surplus and DWL (c = {C[0, col]:.1f})')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()

plt.savefig("../../Apps/Overleaf/14.271_PS_3/figures/mon_comparison_slice_c.png")
plt.show()


# --- Illustrative example for CS ---
# Plot demand function, monopoly price and welfare
delta = 4.5
a = 1
c = 1

p_vals = np.linspace(0, 10, 200)  # range of prices
# Compute values for logit
q_vals_logit = 1 - F(a * p_vals - delta)
p_monopoly_logit = bisection(c, delta/a * 2, foc)
q_monopoly_logit = 1 - F(a * p_monopoly_logit - delta)
q_competitive = 1 - F(a * c - delta)

# Compute values for negative logit
q_vals_neglogit = F(delta - a * p_vals)
p_monopoly_neglogit = bisection(c, delta/a * 2, foc_neg)
q_monopoly_neglogit = F(delta - a * p_monopoly_neglogit)
q_competitive_neg = F(delta - a * c)

plt.figure(figsize=(10, 7))
plt.plot(q_vals_logit, p_vals, label='Logit Inverse Demand', color='blue', lw=2)
plt.plot(q_vals_neglogit, p_vals, label='Neg. Logit Inverse Demand', color='red', lw=2)

plt.axhline(p_monopoly_logit, color='blue', linestyle=':', lw=2)
plt.text(0.05, 3, r"$p^\text{mon}_\text{std}$", color='blue', fontsize=12)
plt.axvline(q_monopoly_logit, color='blue', linestyle=':', lw=2)
plt.text(q_monopoly_logit + 0.005, 9, r"$q^\text{mon}_\text{std}$", color='blue', fontsize=12)
plt.axvline(q_competitive, color='blue', linestyle='-.', lw=1)
plt.text(q_competitive + 0.005, 9, r"$q^\text{comp}_\text{std}$", color='blue', fontsize=12)


plt.axhline(p_monopoly_neglogit, color='red', linestyle=':', lw=2)
plt.text(0.05, 4.3, r"$p^\text{mon}_\text{neg}$", color='red', fontsize=12)
plt.axvline(q_monopoly_neglogit, color='red', linestyle=':', lw=2)
plt.text(q_monopoly_neglogit - 0.05, 9, r"$q^\text{mon}_\text{neg}$", color='red', fontsize=12)
plt.axvline(q_competitive_neg, color='red', linestyle='-.', lw=1)
plt.text(q_competitive_neg - 0.055, 9, r"$q^\text{comp}_\text{neg}$", color='red', fontsize=12)

plt.axhline(c, color='black', linestyle='-.', lw=1)
plt.text(0.05, c + 0.2, "Comp. price = marg. cost", color='black', fontsize=12)

plt.xlabel("Quantity", fontsize=13)
plt.ylabel("Price", fontsize=13)
#plt.title("Demand & Monopoly Outcomes: Logit vs. Negative Logit", fontsize=15)
plt.legend(fontsize=11)
plt.grid(False)
plt.xlim(0,1.07)
plt.ylim(0,10.01)
plt.tight_layout()

plt.savefig("../../Apps/Overleaf/14.271_PS_3/figures/mon_expl.png")
plt.show()


""" Another example where CS and DWL are shaded """
delta = 4.5  # intrinsic utility of the product
# p^m increases in delta
# for small values of delta there might not be a solution
a = 1    # price sensitivity
# price elasticity increases in a
c = 5.5     # marginal cost

p_vals = np.linspace(0, 100, 40001)  # range of prices
# Compute values for logit
q_vals_logit = 1 - F(a * p_vals - delta)
p_monopoly_logit = bisection(c, delta/a * 2, foc)
q_monopoly_logit = 1 - F(a * p_monopoly_logit - delta)
q_competitive = 1 - F(a * c - delta)

# Compute values for negative logit
q_vals_neglogit = F(delta - a * p_vals)
p_monopoly_neglogit = bisection(c, delta/a * 2, foc_neg)
q_monopoly_neglogit = F(delta - a * p_monopoly_neglogit)
q_competitive_neg = F(delta - a * c)

plt.figure(figsize=(10, 7))

plt.plot(q_vals_logit, p_vals, label='Logit Inverse Demand', color='blue', lw=2)
plt.plot(q_vals_neglogit, p_vals, label='Neg. Logit Inverse Demand', color='red', lw=2)

plt.axhline(p_monopoly_logit, color='blue', linestyle=':', lw=2)
plt.fill_between(q_vals_logit, p_monopoly_logit, p_vals, where=(p_vals > p_monopoly_logit), hatch='//',color='blue', alpha=0.3, label='CS (Std. Logit)')

plt.axhline(p_monopoly_neglogit, color='red', linestyle=':', lw=2)
plt.fill_between(q_vals_neglogit, p_monopoly_neglogit, p_vals, where=(p_vals > p_monopoly_neglogit), hatch='//',color='red', alpha=0.3, label='CS (Neg. Logit)')

plt.axvline(q_monopoly_logit, color='blue', linestyle=':', lw=2)
plt.fill_betweenx(p_vals, q_monopoly_logit, q_vals_logit, where=(p_vals > c) & (q_vals_logit > q_monopoly_logit), 
                  color='blue', alpha=0.1, hatch='\\', label='DWL (Std. Logit)')

plt.axvline(q_monopoly_neglogit, color='red', linestyle=':', lw=2)
plt.fill_betweenx(p_vals, q_monopoly_neglogit, q_vals_neglogit, where=(p_vals > c) & (q_vals_neglogit > q_monopoly_neglogit), 
                  color='red', alpha=0.1, hatch='\\', label='DWL (Neg. Logit)')

plt.axhline(c, color='black', linestyle='-.', lw=1)

plt.xlabel("Quantity", fontsize=13)
plt.ylabel("Price", fontsize=13)
plt.legend(fontsize=11)
plt.grid(False)
plt.xlim(0,0.25)
plt.ylim(5,9)
plt.tight_layout()

plt.savefig("../../Apps/Overleaf/14.271_PS_3/figures/mon_cw_dwl.png")
plt.show()
