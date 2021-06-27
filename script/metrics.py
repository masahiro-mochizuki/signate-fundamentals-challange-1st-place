def get_scores(t, y, full=True):
    from scipy.stats import spearmanr
    r_high = spearmanr(t[:, 0], y[:, 0], nan_policy="propagate")[0]
    r_low = spearmanr(t[:, 1], y[:, 1], nan_policy="propagate")[0]
    score = (r_high - 1) ** 2 + (r_low - 1) ** 2
    if full:
        return r_high, r_low, score
    else:
        return score