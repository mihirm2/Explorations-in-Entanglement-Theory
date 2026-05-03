import numpy as np
from scipy.optimize import linprog
from math import comb
import matplotlib.pyplot as plt

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

n_table    = list(range(2, 13))
ce_table   = [ce_values[n] for n in n_table]
zeta_table = [compute_zeta(n, ce_clean) for n in n_table]
n_full  = list(range(2, 32))
ce_full = [ce_values[n] for n in n_full]
theory  = [1 - (3/4)**n for n in n_full]

fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(n_full, ce_full,
        color='#2196F3', marker='o', markersize=6,
        linewidth=2.0, label=r'$\mathcal{C}^*(n)$ — LP upper bound (n=2..31)')
ax.plot(n_table, zeta_table,
        color='#F44336', marker='s', markersize=8,
        linewidth=2.0, linestyle='--', label=r'$\zeta(n)$ — GME Threshold (n=2..12)')
ax.plot(n_full, theory,
        color='#FF9800', linewidth=1.5, linestyle=':',
        label=r'$1 - (3/4)^n$ — theoretical scaling')
ax.fill_between(n_table, zeta_table, ce_table,
                alpha=0.12, color='#4CAF50',
                label='GME-certifiable region (n=2..12)')
ax.axvline(x=12.5, color='gray', linewidth=1.2,
           linestyle='-.', alpha=0.7, label='Table I boundary (n=12)')
ax.axhline(y=1.0, color='gray', linewidth=1.0,
           linestyle='--', alpha=0.4)
ax.set_xlabel('Number of Qubits $n$', fontsize=13)
ax.set_ylabel('Concentratable Entanglement', fontsize=13)
ax.set_title('Concentratable Entanglement: $C^*(n)$, $\\zeta(n)$, and Theoretical Scaling',
             fontsize=14, fontweight='bold')
ax.set_xticks(n_full[::2])
ax.set_xlim(1.5, 31.5)
ax.set_ylim(-0.05, 1.08)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/CE_single_graph.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: CE_clean_graph.png")