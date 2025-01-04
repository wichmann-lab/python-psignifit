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

This option sets the proportion correct to correspond to the threshold on the *unscaled* sigmoid. Possible values are 
in the range from 0 to 1, default is 0.5. The default corresponds to 75\% in a 2AFC task (midway between the guess 
rate of 50 % and ceiling performance 100%).

To set it to a different value, for example to 90 %, you'll do

```{code-cell} ipython3
options = {'thresh_PC': .9}
```

Note that this corresponds to a 95 \% in a 2AFC task.

Be aware that even though the interpretation of the threshold parameter changes with different values for `thresh_PC`,
the prior over the threshold remains unchanged, which means that the prior over psychometric functions will be
shifted. If this is not what you intended, please define a custom prior over the threshold.
