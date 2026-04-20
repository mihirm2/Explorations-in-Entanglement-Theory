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
    """
    Solve the LP from the paper (Eq. 8) using the substitution
    z_j = y_j * 3^(n-2j), which maps the problem to:

        minimize   z_0                (= 3^n * y_0, so CE = 1 - z_0)
        subject to sum_j z_j = 1     (z_j >= 0)
                   M * z >= 0        where M[i,j] = K_i(2j;n) * 9^j

    This eliminates all 3^n scaling issues entirely.
    """
    half_n   = n // 2
    num_vars = half_n + 1

    M = np.zeros((n + 1, num_vars))
    for i in range(n + 1):
        for j in range(num_vars):
            M[i, j] = krawtchouk(i, 2*j, n) * (9**j)
    row_max = np.abs(M).max(axis=1)
    row_max[row_max == 0] = 1.0
    M_norm = M / row_max[:, None]
    c_obj      = np.zeros(num_vars)
    c_obj[0]   = 1.0
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
        z0 = result.x[0]
        return 1.0 - z0
    else:
        return None


def compute_zeta(n, ce_values):
    """
    zeta(n) = max_{1<=k<n} [ C*(k) + C*(n-k) - C*(k)*C*(n-k) ]
    This is the GME threshold: states with CE > zeta(n) must be GME.
    """
    zeta = 0.0
    for k in range(1, n):
        ck  = ce_values.get(k, 0.0)
        cnk = ce_values.get(n - k, 0.0)
        val = ck + cnk - ck * cnk
        if val > zeta:
            zeta = val
    return zeta


def bisep_bound(partition, ce_values):
    """
    Recursively compute CE upper bound for a given partition
    using Proposition 1: C(|A>|B>) = C(A) + C(B) - C(A)*C(B)
    e.g. partition=[3,2] means a 3-qubit and 2-qubit subsystem
    """
    if len(partition) == 1:
        return ce_values.get(partition[0], 0.0)
    c1 = ce_values.get(partition[0], 0.0)
    c2 = bisep_bound(partition[1:], ce_values)
    return c1 + c2 - c1 * c2

print("=" * 55)
print("Running LP solver for n = 1 to 31")
print("=" * 55)

ce_values = {}
for n in range(1, 32):
    val = solve_CE_LP(n)
    ce_values[n] = val
    status = f"{val:.15f}" if val is not None else "FAILED"
    print(f"  n = {n:2d}  ->  C*(n) = {status}")

ce_clean = {k: v for k, v in ce_values.items() if v is not None}

print("\n")
print("=" * 65)
print("TABLE I: Maximal CE and GME Threshold (n = 2 to 12)")
print("(Reproduces Table I from the paper)")
print("=" * 65)
print(f"{'n':<6} {'C*(n)':<30} {'zeta(n)':<25}")
print("-" * 65)
for n in range(2, 13):
    ce   = ce_values.get(n)
    zeta = compute_zeta(n, ce_clean)
    if ce is not None:
        print(f"{n:<6} {ce:<30.15f} {zeta:<25.15f}")
    else:
        print(f"{n:<6} {'FAILED':<30} {'N/A'}")
print("=" * 65)


print("\n")
print("=" * 50)
print("FULL RESULTS: C*(n) for n = 2 to 31")
print("(Reproduces Supplemental Table from the paper)")
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


print("\n")
print("=" * 55)
print("SANITY CHECKS")
print("=" * 55)

vals = [ce_values[n] for n in range(2, 13) if ce_values.get(n) is not None]

print(f"C*(n) strictly increasing (n=2..12):   {all(vals[i] < vals[i+1] for i in range(len(vals)-1))}")
print(f"All C*(n) strictly between 0 and 1:    {all(0 < v < 1 for v in vals)}")
ce_31 = ce_values.get(31)
print(f"C*(31) close to 1:                     {ce_31:.15f}" if ce_31 else "C*(31): FAILED")
print(f"zeta(n) < C*(n) for all n=3..12:       {all(compute_zeta(n, ce_clean) < ce_values[n] for n in range(3,13) if ce_values.get(n))}")

print("\nScaling check: 1 - C*(n) vs (3/4)^n")
print(f"{'n':<6} {'1 - C*(n)':<22} {'(3/4)^n':<22} {'ratio'}")
print("-" * 60)
for n in range(2, 13):
    ce = ce_values.get(n)
    if ce is not None:
        one_minus = 1.0 - ce
        tq_n      = (3.0/4.0)**n
        ratio     = one_minus / tq_n
        print(f"{n:<6} {one_minus:<22.12f} {tq_n:<22.12f} {ratio:.6f}")
print("=" * 55)


print("\n")
print("=" * 50)
print("TABLE II: CE Hierarchy for n = 5 qubits")
print("(Reproduces Table II from the paper)")
print("=" * 50)
print(f"{'Structure':<28} {'Max CE'}")
print("-" * 50)

partitions_5 = [
    [5],
    [3, 2],
    [4, 1],
    [2, 2, 1],
    [3, 1, 1],
    [2, 1, 1, 1],
    [1, 1, 1, 1, 1]
]
for partition in partitions_5:
    label = " x ".join(str(p) for p in partition)
    bound = bisep_bound(partition, ce_clean)
    print(f"{label:<28} {bound:.15f}")
print("=" * 50)