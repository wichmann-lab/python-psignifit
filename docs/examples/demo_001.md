---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


# 1. Basic Usage

The Psignifit 101

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
import matplotlib.pyplot as plt
import numpy as np

import psignifit as ps
from psignifit import psigniplot
```

## Save Data in the Right Format

First we need the data in the format (x | nCorrect | total).
As an example we use the following dataset from a 2AFC experiment with 90
trials at each stimulus level. This dataset comes from a simple signal
detection experiment.

<div class="alert alert-danger"><h4>Warning</h4><p>This format differs slightly from the format used in older
    psignifit versions. You can convert your data by using the reformat
    comand. If you are a user of the older psignifits.</p></div>


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
data = np.array([[0.0010, 45.0000, 90.0000], [0.0015, 50.0000, 90.0000],
                 [0.0020, 44.0000, 90.0000], [0.0025, 44.0000, 90.0000],
                 [0.0030, 52.0000, 90.0000], [0.0035, 53.0000, 90.0000],
                 [0.0040, 62.0000, 90.0000], [0.0045, 64.0000, 90.0000],
                 [0.0050, 76.0000, 90.0000], [0.0060, 79.0000, 90.0000],
                 [0.0070, 88.0000, 90.0000], [0.0080, 90.0000, 90.0000],
                 [0.0100, 90.0000, 90.0000]])
```

## Run psignifit with options
Running psignifit fits a sigmoid function to the data.
You obtain a result object, which contains all the information about
the fitted function.

Running psignifit requires to specify the sigmoid and the experiment type.

Here we choose a cumulative Gauss as the sigmoid and a 2-AFC as the paradigm of the experiment.
This sets the guessing rate to .5 and fits the rest of the parameters '''


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
res = ps.psignifit(data, sigmoid='norm', experiment_type='2AFC')

# Alternatively, you could create and pass a dictionary with the options.

config = {
    'sigmoid': 'norm',
    'experiment_type': '2AFC'
}
res = ps.psignifit(data, **config)
```

There are 3 other types of experiments supported out of the box:

 n alternative forces choice: `experiment_type = '2AFC' (or '3AFC'` or ...)
      The guessing rate is known. Fixes the lower asymptote to 1/n and fits the rest.
 Yes/No experiments: `experiment_type = 'yes/no'`
      A free guessing and lapse rate is estimated
 equal asymptote: `experiment_type = 'equal asymptote'`
    As Yes/No, but enforces that guessing and lapsing occur equally often

 Out of the box psignifit supports the following sigmoid functions,
 chosen by `sigmoid_name = ...`:

+++

 ==================== ================================================
 `sigmoid_name = ...` Distribution
 ==================== ================================================
 'norm'               Cumulative gauss distribution. The default.
 'logistic'           Logistic function. Standard alternative.
 'gumbel'             Cumulative gumbel distribution.
                      Asymmetric, with a longer lower tail.
 'rgumbel'            Reversed gumbel distribution. Asymmetric, with a longer upper tail.
 'tdist'              Student t-distribution with df=1 for heavy tailed functions.
 ==================== ================================================

For positive data on a log-scale, we define the 'weibull' sigmoid class. Notice that it is left
to the user to transform the stimulus level in logarithmic space, and the threshold and width
back to linear space. 'weibull' is therefore just an alias for 'gumbel'.

 ==================== ================================================
 'weibull'            Weibull distribution.
 ==================== ================================================

 Find plots of these sigmoids in the `Sigmoid Demo <sphx_glr_generated_examples_plot_all_sigmoids.py>`.

 There are many other options you can set. You find
 them in `Demo 2 <sphx_glr_generated_examples_demo_002.py>`.

+++

## Visualize the Results

For example you can use the result object res to plot your psychometric
function with the data:


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
psigniplot.plot_psychometric_function(res)
```
