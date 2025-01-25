import numpy as np
import matplotlib.pyplot as plt

from psignifit import sigmoids

sigmoid = sigmoids.sigmoid_by_name('norm', PC=0.5, alpha=0.05)


threshold = 0.5
width = 0.5
gamma = 0.0
lambda_ = 0.0

width_alpha = 0.05


gray = '#888888'

x = np.linspace(1e-12, 1-1e-12, num=10000)
y = sigmoid(x, threshold, width, gamma, lambda_)


fig, ax = plt.subplots(1)
ax.plot(x, y, color='k', linewidth=3)
ax.spines[['top', 'right']].set_visible(False)


# threshold lines
ax.vlines(x=threshold, ymin=0, ymax=0.5, color='k', linestyle='-')
ax.hlines(y=0.5, xmin=x.min(), xmax=threshold, color='k', linestyle='-')


# width dashed lines
x2 = sigmoid.inverse(1-width_alpha, threshold, width)
x1 = sigmoid.inverse(width_alpha , threshold, width)

ax.vlines(x=x1, ymin=0, ymax=sigmoid(x1, threshold, width, gamma, lambda_),
          color=gray, linestyle='--')
ax.vlines(x=x2, ymin=0, ymax=sigmoid(x2, threshold, width, gamma, lambda_),
          color=gray, linestyle='--')

ax.hlines(y=1-width_alpha, xmin=0, xmax=x2, color='k', linestyle='--')
ax.hlines(y=width_alpha, xmin=0, xmax=x1, color='k', linestyle='--')


# annotate threshold
ax.annotate('threshold', xy=(0.5, -0.05), xytext=(0.5, -0.05),
            xycoords='axes fraction', textcoords='axes fraction',
            ha='center', va='center', fontsize=10)

# annotate threshold_PC
ax.annotate('thresh_PC', xy=(-0.05, 0.5), xytext=(-0.015, 0.5),
            xycoords='axes fraction', textcoords='axes fraction',
            ha='right', va='center', fontsize=10)


# annotate width
arrow_linecolor = 'black'
arrow_linewidth = 1

ax.annotate('', xy=(x1, -0.12), xytext=(x2, -0.12), xycoords='axes fraction',
            textcoords='axes fraction',
            arrowprops={'arrowstyle': '<|-|>',
                        'color': arrow_linecolor,
                        'linewidth': arrow_linewidth})

ax.annotate('width', xy=(x2, -0.12), xytext=(x2, -0.12),
            xycoords='axes fraction', textcoords='axes fraction',
            ha='left', va='center', fontsize=10)


# annotate gamma
yoffset = -0.06


ax.annotate('', xy=(yoffset, -0.015), xytext=(yoffset, width_alpha+0.015),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops={'arrowstyle': '<|-|>',
                        'color': arrow_linecolor,
                        'linewidth': arrow_linewidth})

ax.annotate('width_alpha', xy=(yoffset-0.025, 0),
            xytext=(yoffset-0.025, width_alpha/2), xycoords='axes fraction',
            textcoords='axes fraction', rotation=90, ha='center', va='center',
            fontsize=10)

# annotate lambda
ax.annotate('', xy=(yoffset, 1-width_alpha -0.02), xytext=(yoffset, 1+0.02),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops={'arrowstyle': '<|-|>',
                        'color': arrow_linecolor,
                        'linewidth': arrow_linewidth})


ax.annotate('width_alpha ', xy=(yoffset-0.025, width_alpha),
            xytext=(yoffset-0.025, (1-width_alpha /2)),
            xycoords='axes fraction', textcoords='axes fraction',
            rotation=90,
            ha='center', va='center', fontsize=10)


ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["0", "1"], fontsize=14)
ax.set_yticks([0, 1])
ax.set_yticklabels(["0", "1"], fontsize=14)
ax.set_ylabel('Proportion correct', fontsize=14)
ax.set_xlabel('Stimulus level', fontsize=14)
ax.xaxis.set_label_coords(.5, -.2)
ax.yaxis.set_label_coords(-.175, .5)

plt.show()

fig.savefig('sigmoid_and_params_advanced.png', bbox_inches='tight')
