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
This documentation page is still work in progress! Some information might be outdated.
```

*Psignifit* defines the threshold as the point where the unscaled
sigmoid is 0.5, i.e. half-way up its range. 
Sometimes one wants to calculate thresholds for another percent correct level. 

You can change this default by setting the option `threshPC` with the value
of the *proportion correct* you want. For example,

```
options['thresh_PC']   = 0.9
```

will set the threshold value at 0.9 on the unscaled sigmoid.


