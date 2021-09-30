import numpy as np

from psignifit._utils import check_data


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
