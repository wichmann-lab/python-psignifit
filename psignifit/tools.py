import numpy as np
from scipy import stats

from ._utils import check_data
from .sigmoids import sigmoid_by_name
from ._utils import cast_np_scalar

def pool_blocks(data: np.ndarray, max_tol=0, max_gap=np.inf, max_length=np.inf):
    """ Pool trials by stimulus level.

    Pool together trials which differ at most `max_tol`, are separated by maximally `max_gap` trials from other
    levels, and are maximum `max_length` trials apart overall.

    Args:
        data: Array of integer triplets (stimulus level, number correct, number trials)
        max_tol: Maximum difference in stimulus level in each pool of trials. Default: 0.
        max_gap: Maximum trial gap of other stimulus levels to perform pooling. Default: infinity.
        max_length: Maximum trial distance to pool. Default: infinity.
    Returns:
        Pooled data: Array of integer triplets (stimulus level, number correct, number trials),
                     where stimulus level is averaged and correct trial counts are summed over
                     the pooled data.
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


def psychometric_with_eta(stimulus_level, threshold, width, gamma, lambda_,
                          sigmoid, eta, random_state=None):
    """ Psychometric function with overdispersion.

    This is a convenience function used mostly for testing and demos. It first computes proportion correct values
    for a  given psychometric function type, specified by name, and then adds some additional noise to the data,
    so that its variance is compatible with the overdispersion parameter `eta`.

    See Section 2.2 in Schuett, Harmeling, Macke and Wichmann (2016).

    Args:
        stimulus_level: Values of the stimulus value
        threshold: Threshold of the psychometric function
        width: Width of the psychometric function
        gamma: Guess rate
        lambda_: Lapse rate
        sigmoid: A callable, for example a sigmoid function as returned by Configuration.make_sigmoid(), or
                 a name of the sigmoid function to use. See `psignifit.sigmoids.ALL_SIGMOID_NAMES` for the list of
                 available sigmoids
        eta: Overdispersion parameter
        random_state: Random state used to generate the additional variance in the data. If None, NumPy's default
            random number generator is used.
    Returns:
        psi: Proportion correct values for each stimulus level
    """

    if random_state is None:
        random_state = np.random.default_rng()

    if isinstance(sigmoid, str):
        sigmoid = sigmoid_by_name(sigmoid)

    psi = sigmoid(stimulus_level, threshold=threshold, width=width, gamma=gamma, lambd=lambda_)

    new_psi = []
    for p in np.atleast_1d(psi):
        a = ((1/eta**2) - 1) * p
        b = ((1/eta**2) - 1) * (1 - p)
        noised_p = stats.beta.rvs(a=a, b=b, size=1, random_state=random_state)
        new_psi.append(noised_p)

    return cast_np_scalar(np.array(new_psi).squeeze())
