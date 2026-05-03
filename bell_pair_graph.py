import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

def CE(n, epsilon):
    ep = epsilon * (1 - epsilon)
    return 1 - (1 - ep) ** (n / 2)

fig = plt.figure(figsize=(11, 6))
fig.patch.set_facecolor('#0f1235')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.38,
                       left=0.08, right=0.96, top=0.88, bottom=0.18)

ax1 = fig.add_subplot(gs[0, 0])  
ax2 = fig.add_subplot(gs[0, 1])  
ax3 = fig.add_subplot(gs[1, :])  

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#1a1f4e')
    ax.tick_params(colors='#c7d0f8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a4080')

epsilon_vals = np.linspace(0, 1, 500)
n_vals = np.arange(2, 41, 2)

colors1 = plt.cm.plasma(np.linspace(0.2, 0.9, 5))
for i, n in enumerate([2, 4, 8, 16, 32]):
    ax1.plot(epsilon_vals, CE(n, epsilon_vals), color=colors1[i],
             lw=1.6, label=f'$n={n}$')
ax1.axvline(0.5, color='#06b6d4', lw=1, linestyle='--', alpha=0.7)
ax1.set_xlabel('$\\epsilon$', color='#c7d0f8', fontsize=9)
ax1.set_ylabel('$\\mathcal{C}_{|\\psi\\rangle}([n])$', color='#c7d0f8', fontsize=9)
ax1.set_title('CE vs $\\epsilon$ (fixed $n$)', color='white', fontsize=10, pad=6)
ax1.legend(fontsize=7.5, framealpha=0.2, labelcolor='white',
           facecolor='#1a1f4e', edgecolor='#3a4080')
ax1.annotate('max at $\\epsilon=\\frac{1}{2}$', xy=(0.5, CE(8, 0.5)),
             xytext=(0.62, CE(8, 0.5) - 0.08),
             color='#06b6d4', fontsize=7.5,
             arrowprops=dict(arrowstyle='->', color='#06b6d4', lw=1))

colors2 = plt.cm.cool(np.linspace(0.1, 0.9, 5))
for i, eps in enumerate([0.1, 0.25, 0.5, 0.75, 0.9]):
    label = f'$\\epsilon={eps}$' + (' (max)' if eps == 0.5 else '')
    ax2.plot(n_vals, CE(n_vals, eps), color=colors2[i],
             lw=1.6, marker='o', markersize=3, label=label)
ax2.set_xlabel('$n$', color='#c7d0f8', fontsize=9)
ax2.set_ylabel('$\\mathcal{C}_{|\\psi\\rangle}([n])$', color='#c7d0f8', fontsize=9)
ax2.set_title('CE vs $n$ (fixed $\\epsilon$)', color='white', fontsize=10, pad=6)
ax2.legend(fontsize=7.5, framealpha=0.2, labelcolor='white',
           facecolor='#1a1f4e', edgecolor='#3a4080')

ax3.set_title('Interactive: CE vs $\\epsilon$ — drag slider to change $n$',
              color='white', fontsize=10, pad=6)
ax3.set_xlabel('$\\epsilon$', color='#c7d0f8', fontsize=9)
ax3.set_ylabel('$\\mathcal{C}_{|\\psi\\rangle}([n])$', color='#c7d0f8', fontsize=9)

n_init = 6
[line3] = ax3.plot(epsilon_vals, CE(n_init, epsilon_vals),
                   color='#a78bfa', lw=2)
ax3.axvline(0.5, color='#06b6d4', lw=1, linestyle='--', alpha=0.6,
            label='$\\epsilon = \\frac{1}{2}$ (maximum by symmetry)')

max_val = CE(n_init, 0.5)
dot, = ax3.plot([0.5], [max_val], 'o', color='#f97316', zorder=5, markersize=6)
label3 = ax3.text(0.52, max_val + 0.01,
                  f'CE$=$ {max_val:.3f}  at $\\epsilon=\\frac{{1}}{{2}}$',
                  color='#f97316', fontsize=8)
title3 = ax3.set_title(
    f'Interactive: CE vs $\\epsilon$,  $n = {n_init}$ — drag slider to change $n$',
    color='white', fontsize=10, pad=6)
ax3.legend(fontsize=8, framealpha=0.2, labelcolor='white',
           facecolor='#1a1f4e', edgecolor='#3a4080')
ax3.set_ylim(-0.02, 1.05)

ax_slider = fig.add_axes([0.2, 0.04, 0.6, 0.025])
ax_slider.set_facecolor('#252b6b')
slider = Slider(ax_slider, '$n$ (even)', 2, 40, valinit=n_init,
                valstep=2, color='#7c3aed')
slider.label.set_color('white')
slider.valtext.set_color('#a78bfa')

def update(val):
    n = int(slider.val)
    y = CE(n, epsilon_vals)
    line3.set_ydata(y)
    mv = CE(n, 0.5)
    dot.set_ydata([mv])
    label3.set_text(f'CE$=$ {mv:.3f}  at $\\epsilon=\\frac{{1}}{{2}}$')
    label3.set_y(mv + 0.01)
    title3.set_text(
        f'Interactive: CE vs $\\epsilon$,  $n = {n}$ — drag slider to change $n$')
    fig.canvas.draw_idle()

slider.on_changed(update)

fig.suptitle(
    'Concentratable Entanglement of $n/2$ Weakly Entangled Bell Pairs\n'
    r'$\mathcal{C}_{|\psi\rangle}([n]) = 1-(1-\epsilon(1-\epsilon))^{n/2}$',
    color='white', fontsize=12, y=0.97)

plt.show()