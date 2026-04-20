import numpy as np
from scipy.optimize import linprog
from math import comb

def krawtchouk(i, j, n):
    """Quaternary Krawtchouk polynomial K_i(j; n)"""
    result = 0
    for s in range(i + 1):
        if s > j or (i - s) > (n - j) or (i - s) < 0:
            continue
        result += (-1)**s * 3**(i - s) * comb(j, s) * comb(n - j, i - s)
    return result

def solve_CE_LP(n):
    half_n   = n // 2
    num_vars = half_n + 1

    M = np.zeros((n + 1, num_vars))
    for i in range(n + 1):
        for j in range(num_vars):
            M[i, j] = krawtchouk(i, 2*j, n) * (9**j)

    row_max = np.abs(M).max(axis=1)
    row_max[row_max == 0] = 1.0
    M_norm = M / row_max[:, None]

    c_obj    = np.zeros(num_vars)
    c_obj[0] = 1.0

    A_ub = -M_norm
    b_ub = np.zeros(n + 1)

    A_eq = np.ones((1, num_vars))
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * num_vars

    result = linprog(
        c_obj,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method='highs'
    )

    if result.success:
        return 1.0 - result.x[0]
    else:
        return None

def compute_zeta(n, ce_values):
    """
    GME threshold: zeta(n) = max_{1<=k<n} C*(k) + C*(n-k) - C*(k)*C*(n-k)
    Any state with CE > zeta(n) must be genuinely multipartite entangled.
    """
    zeta = 0.0
    for k in range(1, n):
        ck  = ce_values.get(k, 0.0)
        cnk = ce_values.get(n - k, 0.0)
        val = ck + cnk - ck * cnk
        if val > zeta:
            zeta = val
    return zeta

ce_values = {}
for n in range(1, 32):
    ce_values[n] = solve_CE_LP(n)

ce_clean = {k: v for k, v in ce_values.items() if v is not None}

print("=" * 60)
print("TABLE I: Maximal CE and GME Threshold (n = 2 to 12)")
print("=" * 60)
print(f"{'n':<6} {'C*(n)':<28} {'zeta(n)':<25}")
print("-" * 60)
for n in range(2, 13):
    ce   = ce_values.get(n)
    zeta = compute_zeta(n, ce_clean)
    if ce is not None:
        print(f"{n:<6} {ce:<28.15f} {zeta:<25.15f}")
    else:
        print(f"{n:<6} {'FAILED':<28} {'N/A'}")
print("=" * 60)

print("\n")
print("=" * 50)
print("C*(n) Upper Bound for n = 2 to 31")
print("=" * 50)
print(f"{'n':<6} {'C*(n) upper bound'}")
print("-" * 50)
for n in range(2, 32):
    ce = ce_values.get(n)
    if ce is not None:
        print(f"{n:<6} {ce:.15f}")
    else:
        print(f"{n:<6} FAILED")
print("=" * 50)