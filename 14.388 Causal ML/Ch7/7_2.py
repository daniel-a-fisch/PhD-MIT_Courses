# ============================================
# Pearl DAG Exercise 7.2
# ============================================

from pgmpy.models import DiscreteBayesianNetwork
from itertools import combinations

# -----------------------------
# 1. Define original DAG
# -----------------------------
edges = [
    ("Z1", "X1"),
    ("X1", "D"),
    ("D", "M"),
    ("M", "Y"),
    ("Z2", "X3"),
    ("X3", "Y"),
    ("Z1", "X2"),
    ("Z2", "X2"),
    ("X2", "D"),
    ("X2", "Y"),
]

model = DiscreteBayesianNetwork(edges)


# -----------------------------
# Utility: independence checker
# -----------------------------
def independent(model, X, Y, given=None):
    given = given or []
    return not model.is_dconnected(X, Y, observed=given)


def print_test(title, result):
    print(f"{title}: {'TRUE (independent)' if result else 'FALSE (dependent)'}")


def minimal_sets(conditioning_sets):
    """Return subset-minimal conditioning sets."""
    minimal = []
    ordered = sorted(conditioning_sets, key=lambda s: (len(s), tuple(sorted(s))))

    for candidate in ordered:
        if not any(existing.issubset(candidate) for existing in minimal):
            minimal.append(candidate)

    return [tuple(sorted(s)) for s in minimal]


def all_pairwise_conditional_independences(model, observed_nodes):
    observed_nodes = sorted(observed_nodes)
    independences = []

    for X, Y in combinations(observed_nodes, 2):
        candidates = [node for node in observed_nodes if node not in {X, Y}]
        conditioning_sets = []

        for size in range(len(candidates) + 1):
            for given in combinations(candidates, size):
                if independent(model, X, Y, list(given)):
                    conditioning_sets.append(frozenset(given))

        for given in minimal_sets(conditioning_sets):
            independences.append((X, Y, given))

    return independences


def print_independences(model, observed_nodes):
    results = all_pairwise_conditional_independences(model, observed_nodes)
    if not results:
        print("No pairwise conditional independences found.")
        return

    for X, Y, given in results:
        if given:
            print(f"{X} ⟂ {Y} | {', '.join(given)}")
        else:
            print(f"{X} ⟂ {Y}")


# ============================================
# Q1: All variables observed
# ============================================

print("\n================ Q1 =================")
print("Testable implications (d-separation):\n")

print("Unconditional pairwise checks (only return independent):")
for X, Y in combinations(sorted(model.nodes()), 2):
    if independent(model, X, Y):
        print_test(f"{X} ⟂ {Y}", True)

print("\nCollider test (conditioning on X2):")
print_test("Z1 ⟂ Z2 | X2", independent(model, "Z1", "Z2", ["X2"]))

print("\nKey conditional independence:")
print_test("D ⟂ Y | M, X2", independent(model, "D", "Y", ["M", "X2"]))


# ============================================
# Q2: Only {D, Y, X2, M}
# ============================================

print("\n================ Q2 =================")
print("Observed: {D, Y, X2, M}\n")

print("Minimal pairwise independences over all conditioning sets (observed only):")
print_independences(model, ["D", "Y", "X2", "M"])


# ============================================
# Q3: Only {D, Y, X2}
# ============================================

print("\n================ Q3 =================")
print("Observed: {D, Y, X2}\n")

print("Minimal pairwise independences over all conditioning sets (observed only):")
print_independences(model, ["D", "Y", "X2"])


# ============================================
# Q4: All except X2 (latent)
# ============================================

print("\n================ Q4 =================")
print("X2 is unobserved (latent)\n")

observed_q4 = sorted(node for node in model.nodes() if node != "X2")
print("Unconditional pairwise checks among observed nodes:")
for X, Y in combinations(observed_q4, 2):
    if independent(model, X, Y):
        print_test(f"{X} ⟂ {Y}", True)


# ============================================
# Q5: Alternative model (reverse X2 → D)
# ============================================

print("\n================ Q5 =================")
print("Compare original vs alternative model\n")

edges_alt = [
    ("Z1", "X1"),
    ("X1", "D"),
    ("D", "M"),
    ("M", "Y"),
    ("Z2", "X3"),
    ("X3", "Y"),
    ("Z1", "X2"),
    ("Z2", "X2"),
    ("D", "X2"),  # reversed edge
    ("X2", "Y"),
]

model_alt = DiscreteBayesianNetwork(edges_alt)


def independent_alt(X, Y, given=None):
    given = given or []
    return not model_alt.is_dconnected(X, Y, observed=given)


print("Test: Z1 ⟂ Z2 | D\n")

print_test("Original model", independent(model, "Z1", "Z2", ["D"]))
print_test("Alternative model", independent_alt("Z1", "Z2", ["D"]))
