import numpy as np

import psignifit as ps
from psignifit.utils import my_norminv, my_t1icdf


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
           theta(1) = threshold -> threshold
           theta(2) = width     -> standard deviation
     logistic
           theta(1) = threshold -> threshold
           theta(2) = width     -> slope at threshold
     Weibull/weibull
           theta(1) = threshold -> scale
           theta(2) = width     -> shape parameter
     gumbel & rgumbel distributions
           theta(1) = threshold -> mode
           theta(2) = width     -> scale
     tdist
           theta(1) = threshold -> threshold/mean
           theta(2) = width     -> standard deviation

     For negative slope sigmoids we return the exact same parameters as for
     the positive ones.
    """

    # if theta is a result dict from psignifit, obtain parameters from there
    if type(theta) == dict:
        widthalpha = theta["options"]["widthalpha"]
        sigmoid_name = theta["options"]["sigmoidName"]

        if theta["option"]["threshPC"] != 0.5:
            if theta["options"]["logspace"]:
                theta["Fit"][0] = np.log(ps.getThreshold(theta, .5, True))
            else:
                theta["Fit"][0] = ps.getThreshold(theta, .5, True)

        theta = theta["Fit"]

    # TODO: does this work for both a list of parameters AND theta["Fit"]
    if sigmoid_name in ["norm", "gauss", "neg_norm", "neg_gauss"]:
        theta[0] = theta[0]
        C = my_norminv(1 - widthalpha, 0, 1) - my_norminv(widthalpha, 0, 1)
        theta[1] = theta[1] / C
    elif sigmoid_name in ["logistic", "neg_logistic"]:
        pass
    elif sigmoid_name in ["Weibull", "weibull", "neg_Weibull", "neg_weibull"]:
        pass
    elif sigmoid_name in ["gumbel", "neg_gumbel"]:
        pass
    elif sigmoid_name in ["rgumbel", "neg_rgumbel"]:
        pass
    elif sigmoid_name in ["tdist", "student", "heavytail", "neg_tist", "neg_student", "neg_heavytail"]:
        pass
    else:
        raise ValueError("Please specify a valid sigmoid name, either explicitly or via result dict")


if __name__ == "__main__":
    pass
