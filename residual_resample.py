import numpy as np

def residual_resample(in_index, weights):
    """
    Residual Resampling for Particle Filter.

    Parameters
    ----------
    in_index : array-like
        Input particle indices (1D array of integers).
    weights : array-like
        Normalized importance weights (must sum to 1).

    Returns
    -------
    out_index : ndarray
        Resampled indices of particles.
    """
    S = len(weights)
    out_index = np.zeros(S, dtype=int)

    # Integer part
    weights_res = S * np.array(weights)
    N_kind = np.floor(weights_res).astype(int)

    # Number of residual samples to generate
    N_res = S - np.sum(N_kind)

    if N_res > 0:
        weights_residual = (weights_res - N_kind) / N_res
        cum_dist = np.cumsum(weights_residual)

        # Stratified sampling approach for better diversity
        u = np.flip(np.cumprod(np.random.rand(N_res) ** (1.0 / np.arange(N_res, 0, -1))))

        for val in u:
            j = np.searchsorted(cum_dist, val)
            N_kind[j] += 1

    # Build the output index array
    index = 0
    for i in range(S):
        count = N_kind[i]
        if count > 0:
            out_index[index:index+count] = in_index[i]
            index += count

    return out_index