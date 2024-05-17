import numpy as np
import scipy

batch_based_metrics = {
    "neg_mse": lambda y, y_hat: np.mean((y - y_hat) ** 2),
    # (Protected) inverse mean squared error
    "inv_mse": lambda y, y_hat: 1 / (1 + np.mean((y - y_hat) ** 2)),
    # Pearson correlation coefficient       # Range: [0, 1]
    "pearson": lambda y, y_hat: scipy.stats.pearsonr(y, y_hat)[0],
    # Spearman correlation coefficient      # Range: [0, 1]
    "spearman": lambda y, y_hat: scipy.stats.spearmanr(y, y_hat)[0],
}


def compute_disagreement_score(arrays, metric_function):
    """
    arrays shape: num_odes, num_traj*time_step* num_variables
    """
    metric = batch_based_metrics[metric_function]
    num_odes, total_traj = arrays.shape
    final_scores = np.zeros(shape=(num_odes, num_odes))
    # Calculate pairwise metric values
    for i in range(num_odes):
        for j in range(i + 1, num_odes):
            score = metric(arrays[i], arrays[j])
            final_scores[i, j] = score
            final_scores[j, i] = score  # The matrix is symmetric

    # Sum only the upper triangle (excluding the diagonal) of the final_scores matrix
    upper_triangle_sum = np.sum(np.triu(final_scores, k=1)) / (num_odes * num_odes)

    return upper_triangle_sum
