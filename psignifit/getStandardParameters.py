import numpy as np

import psignifit as ps
from .utils import my_norminv, my_t1icdf


def getStandardParameters(theta, sigmoid_name=None, widthalpha=.05):
    """ this function transforms a parameter given in threshold, width to a
     standard parameterization for comparison purposes

     theta = parameter
     sigmoid_name  = Name of the Sigmoid

     if you changed the widthalpha you should pass it as an additional
     argument

     Alternatively you can pass the result struct instead of the parameter
     values. Then type and widthalpha are set automatically.

     norm/gauss/normal
           theta[0] = threshold -> threshold
           theta[1] = width     -> standard deviation
     logistic
           theta[0] = threshold -> threshold
           theta[1] = width     -> slope at threshold
     Weibull/weibull
           theta[0] = threshold -> scale
           theta[1] = width     -> shape parameter
     gumbel & rgumbel distributions
           theta[0] = threshold -> mode
           theta[1] = width     -> scale
     tdist
           theta[0] = threshold -> threshold/mean
           theta[1] = width     -> standard deviation

     For negative slope sigmoids we return the exact same parameters as for
     the positive ones.
    """

    # if theta is a result dict from psignifit, obtain parameters from there
    if isinstance(theta, dict):
        widthalpha = theta["options"]["widthalpha"]
        sigmoid_name = theta["options"]["sigmoidName"]

        if theta["options"]["threshPC"] != 0.5:
            if theta["options"]["logspace"]:
                theta["Fit"][0] = np.log(ps.getThreshold(theta, .5, True))
            else:
                theta["Fit"][0] = ps.getThreshold(theta, .5, True)

        theta = theta["Fit"].copy()

    elif isinstance(theta, np.ndarray):  # theta is assumed to be a numpy array
        theta = theta.copy()
    else:
        raise ValueError("theta must be either a dict (result from psignifit) or a numpy array.")

    if sigmoid_name in ["norm", "gauss", "neg_norm", "neg_gauss"]:
        theta[0] = theta[0]
        C = my_norminv(1 - widthalpha, 0, 1) - my_norminv(widthalpha, 0, 1)
        theta[1] = theta[1] / C
    elif sigmoid_name in ["logistic", "neg_logistic"]:
        theta[0] = theta[0]
        theta[1] = 2 * np.log(1. / widthalpha - 1) / theta[1]
    elif sigmoid_name in ["Weibull", "weibull", "neg_Weibull", "neg_weibull"]:
        C = np.log(-np.log(widthalpha)) - np.log(-np.log(1 - widthalpha))
        shape = C / theta[1]
        scale = np.exp(C / theta[1] * (-theta[0]) + np.log(-np.log(.5)))
        scale = np.exp(np.log(1 / scale) / shape)  # Wikipediascale
        theta[0] = scale
        theta[1] = shape
    elif sigmoid_name in ["gumbel", "neg_gumbel"]:
        # note that gumbel and reversed gumbel definitions are sometimes swapped
        # and sometimes called extreme value distributions

        C = np.log(-np.log(widthalpha)) - np.log(-np.log(1 - widthalpha))
        theta[1] = theta[1] / C
        theta[0] = theta[0] - theta[1] * np.log(-np.log(.5))
    elif sigmoid_name in ["rgumbel", "neg_rgumbel"]:
        C = np.log(-np.log(1 - widthalpha)) - np.log(-np.log(widthalpha))
        theta[1] = -theta[1] / C
        theta[0] = theta[0] + theta[1] * np.log(-np.log(.5))
    elif sigmoid_name in ["tdist", "student", "heavytail", "neg_tist", "neg_student", "neg_heavytail"]:
        C = my_t1icdf(1 - widthalpha) - my_t1icdf(widthalpha)
        theta[0] = theta[0]
        theta[1] = theta[1] / C
    else:
        raise ValueError("Please specify a valid sigmoid name, either explicitly or via result dict")

    return theta
