import numpy as np
import matplotlib.pyplot as plt

from psignifit import sigmoids

sigmoid = sigmoids.sigmoid_by_name('norm', PC=0.5, alpha=0.05)


threshold = 0.5
width = 0.5
gamma = 0.1
lambda_ = 0.1

gray = '#888888'

x = np.linspace(1e-12, 1-1e-12, num=10000)
y = sigmoid(x, threshold, width, gamma, lambda_)


fig, ax = plt.subplots(1)
ax.plot(x, y, color='k', linewidth=3)
ax.spines[['top', 'right']].set_visible(False)


# threshold lines
ax.vlines(x=threshold, ymin=0, ymax=0.5, color='k', linestyle='-')
ax.hlines(y=0.5, xmin=x.min(), xmax=threshold, color='k', linestyle='-')

# gamma and lambda dashed lines
ax.hlines(y=(1-lambda_), xmin=x.min()-0.05, xmax=x.max(), color='k', linestyle='--')
ax.hlines(y=gamma, xmin=x.min()-0.05, xmax=x.max(), color='k', linestyle='--')


# width dashed lines
x2 = sigmoid.inverse(0.95, threshold, width)
x1 = sigmoid.inverse(0.05, threshold, width)

ax.vlines(x=x1, ymin=0, ymax=sigmoid(x1, threshold, width, gamma, lambda_),
          color=gray, linestyle='--')
ax.vlines(x=x2, ymin=0, ymax=sigmoid(x2, threshold, width, gamma, lambda_),
          color=gray, linestyle='--')

#ax.hlines(y=0.95-lambda_, xmin=0, xmax=x2, color=gray, linestyle='--')
#ax.hlines(y=0.05+gamma, xmin=0, xmax=x1, color=gray, linestyle='--')

#ax.annotate('0.05+gamma', xy=(0.0, 0.05+gamma), xytext=(-0.01, 0.05+gamma),
#            xycoords='axes fraction', textcoords='axes fraction',
#            ha='right', va='center', fontsize=10)


# annotate threshold
ax.annotate('threshold', xy=(0.5, -0.05), xytext=(0.5, -0.05),
            xycoords='axes fraction', textcoords='axes fraction',
            ha='center', va='center', fontsize=10)

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


ax.annotate('', xy=(yoffset, -0.01), xytext=(yoffset, gamma+0.01),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops={'arrowstyle': '<|-|>',
                        'color': arrow_linecolor,
                        'linewidth': arrow_linewidth})

ax.annotate('gamma', xy=(yoffset-0.025, 0),
            xytext=(yoffset-0.025, gamma/2), xycoords='axes fraction',
            textcoords='axes fraction', rotation=90, ha='center', va='center',
            fontsize=10)

# annotate lambda
ax.annotate('', xy=(yoffset, 1-lambda_-0.01), xytext=(yoffset, 1+0.01),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops={'arrowstyle': '<|-|>',
                        'color': arrow_linecolor,
                        'linewidth': arrow_linewidth})


ax.annotate('lambda', xy=(yoffset-0.025, 0),
            xytext=(yoffset-0.025, (1-lambda_/2)),
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
ax.yaxis.set_label_coords(-.15, .5)

fig.savefig('sigmoid_and_params.png', bbox_inches='tight')
