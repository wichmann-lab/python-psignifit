---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Get standard parameters

In psignifit 4 we chose to re-parametrized all sigmoids to the common parameter space spanned by threshold and width. To compare these parameters to values from the literature, older fits, etc. we provide a function transforming our parameters to the common standard parametrizations. It is implemented as a method on the results object, like this

```
result.standard_parameter_estimate()
```

For example, consider a Gaussian sigmoid with alpha=0.05, PC=0.5 fitted. The parameters we would get out, threshold and width, correspond to the standard parameters loc (mean) and scale (standard deviation) in the following way:

```
expected_loc = threshold
# 1.644853626951472 is the normal PPF at alpha=0.95
expected_scale = width / (2 * 1.644853626951472)
```
For different sigmoids these transforms will differ. The `standard_parameter_estimate` method computes the standard values for any of the different sigmoids.

```
loc, scale = result.standard_parameters_estimate()
```

