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

# Change the threshold percentage correct 

```{warning}
The option to change the parameter 'thresh_PC' is so far only implemented in the MATLAB version of psignifit.
In this python version they are still work in progress and can not be changed from the default.
```

This option sets the proportion correct to correspond to the threshold on the *unscaled* sigmoid. Possible values are in the range from 0 to 1, default is 0.5. The default corresponds to 75\% in a 2AFC task (midway between the guess rate of 50 % and ceiling performance 100%).

To set it to a different value, for example to 90 %, you'll do

```{code-cell} ipython3
options = {'thresh_PC': .9}
```

Note that this corresponds to a 95 \% in a 2AFC task.
