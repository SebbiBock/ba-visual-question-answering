import numpy as np
import numpy.random as random
from scipy.stats import spearmanr


def convert_saliency_map_to_density(saliency_map, minimum_value=0.0):
    if saliency_map.min() < 0:
        saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map + minimum_value

    saliency_map_sum = saliency_map.sum()
    if saliency_map_sum:
        saliency_map = saliency_map / saliency_map_sum
    else:
        saliency_map[:] = 1.0
        saliency_map /= saliency_map.sum()

    return saliency_map


def probabilistic_image_based_kl_divergence(logp1, logp2, log_regularization=0, quotient_regularization=0):
    if log_regularization or quotient_regularization:
        return (np.exp(logp2) * np.log(log_regularization + np.exp(logp2) / (np.exp(logp1) + quotient_regularization))).sum()
    else:
        return (np.exp(logp2) * (logp2 - logp1)).sum()


def image_based_kl_divergence(saliency_map_1, saliency_map_2, minimum_value=1e-20, log_regularization=0, quotient_regularization=0):
    """ KLDiv. Function is not symmetric. saliency_map_2 is treated as empirical saliency map. """
    log_density_1 = np.log(convert_saliency_map_to_density(saliency_map_1, minimum_value=minimum_value))
    log_density_2 = np.log(convert_saliency_map_to_density(saliency_map_2, minimum_value=minimum_value))

    return probabilistic_image_based_kl_divergence(log_density_1, log_density_2, log_regularization=log_regularization, quotient_regularization=quotient_regularization)


def new_kl_div(saliency_map_1, saliency_map_2):
    """
        Compute image-based KL divergence with same hyperparameters as in Tuebingen/MIT Saliency Benchmark.

        TAKEN FROM MIT / TUEBINGEN SALIENCY BENCHMARK
    """
    return image_based_kl_divergence(
        saliency_map_1,
        saliency_map_2,
        minimum_value=0,
        log_regularization=2.2204e-16,
        quotient_regularization=2.2204e-16
    )


def kl_divergence(machine_map: np.array, human_map: np.array, epsilon=1e-5):
    """
        Implement the Kullback-Leibler Divergence as a measure of dissimilarity
        between two heatmaps. If both heatmaps are considered to be equal, the
        KL-Div value is zero. The measure is only bound to be >=0. The epsilon
        is used to ensure numerical stability if some value is zero.

        NOTE: THE KL-DIVERGENCE IS NOT SYMMETRIC!

        DISTRIBUTION BASED METRIC

        :param machine_map: The Transformer Heatmap, scaled to display size of the image
        :param human_map: The Human Heatmap, created through gaussian smoothing
        :return: The KL-Div score
    """

    # Assert same shape
    assert machine_map.shape == human_map.shape

    # FLatten
    sm = machine_map.flatten()
    fm = human_map.flatten()

    # Normalize
    sm = sm / sm.sum()
    fm = fm / fm.sum()

    # Add epsilon for numerical stability
    sm += epsilon
    fm += epsilon

    # Compute and return the KL-Divergence
    return (fm * np.log2(fm / sm)).sum()


def rank_correlation(machine_map: np.array, human_map: np.array):
    """
        Implement the spearman rank-order correlation as a measure of similarity between two heatmaps.
        The rank correlation takes values between [-1, 1], where 1 and -1 indicate total correlation
        (positive / negative), where 0 indicates no correlation.

        NOTE: The ordering of the ranking does not matter apparently

        DISTRIBUTION BASED METHOD

        :param machine_map: The Transformer Heatmap in its original size
        :param human_map: The Human Heatmap, scaled down to the size of the transformer heatmap
        :return: The KL-Div score
    """

    # Assert same shape
    assert machine_map.shape == human_map.shape

    # Flatten
    sm = machine_map.flatten()
    fm = human_map.flatten()

    # Normalize
    sm = sm / sm.sum()
    fm = fm / fm.sum()

    return spearmanr(sm, fm).correlation


def auc_borji(saliency_map, fixation_map, n_rep=100, step_size=0.1):
    """

        Implement the Area Under the Curve for the ROC-Curve of a saliency map. This measures how well the saliency map
        of an image predicts the ground truth human fixations on the image. The low-size transformer heatmap is needed,
        since we need to determine whether a point was a fixation. For this, we simply need to check if the human data
        has a value greater or equal to 0.1 assigned to it, meaning that at least one human subject fixated on this point.

        LOCATION BASED METRIC

        TAKEN FROM https://github.com/imatge-upc/saliency-2019-SalBCE.

        :param saliency_map: The Transformer Heatmap in its original size
        :param fixation_map: The Human Heatmap, scaled down to the size of the transformer heatmap
        :param n_rep: The number of random splits to compute
        :param step_size: The step size by which the threshold is to be increased
        :return: The AUC-Borji score, [0-1]
    """

    # Get fixations
    fixation_map = fixation_map >= 0.1

    # If there are no fixation to predict, return 0
    if not np.any(fixation_map):
        print('no fixation to predict')
        return 0

    # Normalize saliency map to have values between [0,1]
    saliency_map = saliency_map / saliency_map.max()

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)

    # For each fixation, sample n_rep values from anywhere on the saliency map
    r = np.random.randint(0, n_pixels, [n_fix, n_rep])
    S_rand = S[r]  # Saliency map values at random locations (including fixated locations!? underestimated)

    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:, rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds) + 2)
        fp = np.zeros(len(thresholds) + 2)
        tp[0] = 0;
        tp[-1] = 1
        fp[0] = 0;
        fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k + 1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k + 1] = np.sum(S_rand[:, rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)

    # Average across random splits and return
    return np.mean(auc)


def nss(saliency_map, fixation_map):
    """
        Implement the normalized scanpath saliency of a saliency map, defined as the mean value of normalize saliency values
        at fixation locations. Larger values imply higher similarity. The low-size transformer heatmap is needed, since we
        need to determine whether a point was a fixation. For this, we simply need to check if the human data has a value
        greater or equal to 0.1 assigned to it, meaning that at least one human subject fixated on this point.

        VALUE BASED METRIC THAT IS USED AS REPRESENTATIVE FOR CLUSTER

        :param saliency_map: The Transformer Heatmap in its original size
        :param fixation_map: The Human Heatmap, scaled down to the size of the transformer heatmap.
        :return: The NSS score
    """

    # Get fixations
    f_map = np.vectorize(lambda x: 1 if x > 0 else 0)(fixation_map) > 0.5

    # Normalize saliency map to have zero mean and unit std
    s_map = (saliency_map - np.mean(saliency_map)) / np.std(saliency_map)

    # Return mean saliency value at fixation locations
    return np.mean(s_map[f_map])


def auc_judd(saliency_map, fixation_map, jitter=True):
    '''

    TAKEN FROM MIT / TUEBINGEN SALIENCY BENCHMARK

    AUC stands for Area Under ROC Curve.
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve is created by sweeping through threshold values
    determined by range of saliency map values at fixation locations.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at all other locations to the total number of possible other locations (non-fixated image pixels).
    AUC=0.5 is chance level.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    jitter : boolean, optional
        If True (default), a small random number would be added to each pixel of the saliency map.
        Jitter saliency maps that come from saliency models that have a lot of zero values.
        If the saliency map is made with a Gaussian then it does not need to be jittered
        as the values vary and there is not a large patch of the same value.
        In fact, jittering breaks the ordering in the small values!
    Returns
    -------
    AUC : float, between [0,1]
    '''

    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(np.vectorize(lambda x: 1 if x > 0 else 0)(fixation_map), copy=False) > 0.5

    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp) # y, x
