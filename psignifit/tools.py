import numpy as np
from scipy import stats

from psignifit._utils import check_data
from psignifit.sigmoids import sigmoid_by_name

def pool_blocks(data: np.ndarray, max_tol=0, max_gap=np.inf, max_length=np.inf):
    """ Pool trials

    Pool trials together which differ at max_tol from the first one
    it finds, are separated by maximally max_gap trials of other levels and
    at max max_length trials appart in general.

    Args:
        data: Array of integer triplets (stimulus level, number correct, number trials)
        max_tol: Maximum difference in stimulus level to pool trials. Default: 0.
        max_gap: Maximum trial gap of other stimulus levels to perform pooling. Default: infinity.
        max_length: Maximum trial distance to pool. Default: infinity.
    Returns:
        Pooled data: Array of integer triplets (stimulus level, number correct, number trials),
                     where stimulus level is averaged and correct/trial numbers are summed over
                     the pooled data rows.
    """
    data = check_data(data)

    ndata = data.shape[0]
    seen = [False] * ndata
    cum_ntrials = [0] + list(data[:, 2].cumsum())

    pool = []
    for i in range(ndata):
        if not seen[i]:
            current = data[i, 0]
            block = []
            gap = 0
            for j in range(i, ndata):
                if (cum_ntrials[j + 1] -
                        cum_ntrials[i]) > max_length or gap > max_gap:
                    break
                level, ncorrect, ntrials = data[j, :]
                if abs(level - current) <= max_tol and not seen[j]:
                    seen[j] = True
                    gap = 0
                    block.append((level * ntrials, ncorrect, ntrials))
                else:
                    gap += ntrials

            level, ncorrect, ntrials = np.sum(block, axis=0)
            pool.append((level / ntrials, ncorrect, ntrials))

    return np.array(pool)


def psychometric(stimulus_level, threshold, width, gamma, lambda_, sigmoid_name):
    """ Psychometric function aka proportion correct function.

    This is a convenience function used mostly for testing and demos, to make the intent of creating a psychometric
    function more explicit. The implementation simply combines creating a `Sigmoid` object using the sigmoid name,
    and calling the object to generate the proportion correct values corresponding to `stimulus_level`.

    Parameters:
        stimulus_level: array
          Values of the stimulus value
        threshold: float
            Threshold of the psychometric function
        width: float
            Width of the psychometric function
        gamma: float
            Guess rate
        lambda_: float
            Lapse rate
        sigmoid_name: callable
            Name of sigmoid function to use. See `psignifit.sigmoids.ALL_SIGMOID_NAMES` for the list of available
            sigmoids.

    Returns:
        psi: array
            Proportion correct values for each stimulus level

    """
    sigmoid = sigmoid_by_name(sigmoid_name)
    psi = sigmoid(stimulus_level, threshold=threshold, width=width, gamma=gamma, lambd=lambda_)
    return psi


def psychometric_with_eta(stimulus_level, threshold, width, gamma, lambda_,
                          sigmoid_name, eta, random_state=None):
    """ Psychometric function with overdispersion.

    This is a convenience function used mostly for testing and demos. Just like the function `psychometric`, it
    computes proportion correct values for a given psychometric function type, specified by name. In addition,
    it adds some additional noise to the data, so that its variance is compatible with the overdispersion
    parameter `eta`.

    See Section 2.2 in Schuett, Harmeling, Macke and Wichmann (2016).

    Parameters:
        stimulus_level: array
          Values of the stimulus value
        threshold: float
            Threshold of the psychometric function
        width: float
            Width of the psychometric function
        gamma: float
            Guess rate
        lambda_: float
            Lapse rate
        sigmoid_name: callable
            Name of sigmoid function to use. See `psignifit.sigmoids.ALL_SIGMOID_NAMES` for the list of available
            sigmoids.
        eta: float
            Overdispersion parameter
        random_state: np.RandomState
            Random state used to generate the additional variance in the data.
            If None, NumPy's default random number generator is used.

    Returns:
        psi: array
            Proportion correct values for each stimulus level

    """

    if random_state is None:
        random_state = np.random.default_rng()

    psi = psychometric(stimulus_level, threshold, width, gamma, lambda_, sigmoid_name)
    new_psi = []
    for p in psi:
        a = ((1/eta**2) - 1) * p
        b = ((1/eta**2) - 1) * (1 - p)
        noised_p = stats.beta.rvs(a=a, b=b, size=1, random_state=random_state)
        new_psi.append(noised_p)
    return np.array(new_psi).squeeze()
