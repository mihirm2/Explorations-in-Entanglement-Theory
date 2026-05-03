import numpy as np
import matplotlib.pyplot as plt

def CE(n, epsilon):
    ep = epsilon * (1 - epsilon)
    return 1 - (1 - ep) ** (n / 2)

n_vals = np.arange(2, 62, 2)
epsilons = [0.05, 0.1, 0.2, 0.35, 0.5]
labels = [r'$\epsilon=0.05$', r'$\epsilon=0.1$', r'$\epsilon=0.2$', r'$\epsilon=0.35$', r'$\epsilon=0.5$']
colors = ['#4488cc', '#44aa88', '#dd8833', '#9955bb', '#cc3333']

asymptotes = [1 - (1 - eps*(1-eps))**500 for eps in epsilons]

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

for eps, col, lab, asym in zip(epsilons, colors, labels, asymptotes):
    y = CE(n_vals, eps)
    ax.plot(n_vals, y, color=col, lw=1.8)
    ax.axhline(y=asym, color=col, lw=1.0, linestyle='--', alpha=0.75)

curve_ends = [CE(60, eps) for eps in epsilons]
min_gap = 0.042
label_positions = list(curve_ends)
for i in range(1, len(label_positions)):
    if label_positions[i] - label_positions[i-1] < min_gap:
        label_positions[i] = label_positions[i-1] + min_gap

for col, lab, ly in zip(colors, labels, label_positions):
    ax.annotate(lab, xy=(62, ly), xytext=(63.5, ly),
                color=col, fontsize=8.5, va='center',
                annotation_clip=False)

ax.set_xlabel('number of qubits', fontsize=10, color='#333333')
ax.set_ylabel('CE', fontsize=10, color='#333333')
ax.set_title('CE of Weakly Entangled Bell Pairs vs $n$', fontsize=11, color='#222222')
ax.set_xlim(0, 62)
ax.set_ylim(0.0, 1.08)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#aaaaaa')
ax.spines['bottom'].set_color('#aaaaaa')
ax.tick_params(colors='#555555', labelsize=8)
ax.grid(False)

plt.subplots_adjust(right=0.75)
plt.savefig('ce_plot.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()