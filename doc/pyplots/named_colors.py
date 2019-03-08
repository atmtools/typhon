import matplotlib.pyplot as plt

from typhon.plots import TYPHON_COLORS

fig, ax = plt.subplots(figsize=(8, 5))

# Get height and width
X, Y = fig.get_dpi() * fig.get_size_inches()
h = Y / (len(TYPHON_COLORS) + 1)
w = X

for row, name in enumerate(sorted(TYPHON_COLORS.keys())):
    y = Y - (row * h) - h

    xi_line = w * 0.05
    xf_line = w * 0.25
    xi_text = w * 0.3

    ax.text(xi_text, y, name, fontsize=(h * 0.8),
            horizontalalignment='left',
            verticalalignment='center')

    ax.hlines(y + h * 0.1, xi_line, xf_line,
              color=TYPHON_COLORS[name], linewidth=(h * 0.6))

ax.set_xlim(0, X)
ax.set_ylim(0, Y)
ax.set_axis_off()

fig.subplots_adjust(left=0, right=1,
                    top=1, bottom=0,
                    hspace=0, wspace=0)

plt.show()
