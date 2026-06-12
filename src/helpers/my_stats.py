import numpy as np
from scipy.stats import chi2, rankdata




def wheeler_watson_test(samples):
    """
    Robust Wheeler-Watson two/multi-sample test for circular data.

    Parameters
    ----------
    samples : list of array-like
        samples of angles in radians (can be in [-π, π] or [0, 2π])

    Returns
    -------
    W : float
        Test statistic
    pval : float
        p-value
    df : int
        Degrees of freedom
    """

    # --- Convert to numpy arrays and wrap to [0, 2π)
    samples = [np.mod(np.asarray(s), 2 * np.pi) for s in samples]

    k = len(samples)
    if k < 2:
        raise ValueError("At least two samples are required.")

    # --- Pool all data
    pooled = np.concatenate(samples)
    N = len(pooled)

    # --- Compute ranks (handles ties properly)
    ranks = rankdata(pooled, method="average")

    # --- Convert ranks to circular ranks
    circ_ranks = 2 * np.pi * ranks / N

    # --- Split ranks back into groups
    split_indices = np.cumsum([len(s) for s in samples])[:-1]
    circ_groups = np.split(circ_ranks, split_indices)

    # --- Compute test statistic
    if k == 2:
        n1, n2 = len(samples[0]), len(samples[1])
        C = np.sum(np.cos(circ_groups[0]))
        S = np.sum(np.sin(circ_groups[0]))
        W = 2 * (N - 1) * (C**2 + S**2) / (n1 * n2)
    else:
        W = 0.0
        for g in circ_groups:
            C = np.sum(np.cos(g))
            S = np.sum(np.sin(g))
            W += (C**2 + S**2) / len(g)
        W *= 2.0

    df = 2 * (k - 1)
    pval = chi2.sf(W, df)

    return W, pval, df