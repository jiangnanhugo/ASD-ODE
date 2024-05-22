import numpy as np
import scipy
from sklearn.cluster import KMeans

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


#####
def deep_coreset(data, sample_size=20, n_clusters=5):
    """
    data of shape  (n_samples, n_features)
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)

    # Get cluster labels for each instance
    cluster_labels = kmeans.labels_

    # Step 2: Sample Diverse Samples from Each Cluster
    # Sample size from each cluster
    diverse_samples = []

    # Iterate through each cluster
    for cluster_id in range(n_clusters):
        # Extract instances belonging to the current cluster
        cluster_data = data[cluster_labels == cluster_id]

        # If the cluster has fewer data points than the sample size, sample all instances
        if len(cluster_data) <= sample_size:
            cluster_samples = cluster_data
        else:
            # Compute pairwise distances between instances in the cluster
            distances = np.linalg.norm(cluster_data[:, None] - cluster_data, axis=-1)

            # Compute diversity score for each instance
            diversity_scores = np.sum(distances, axis=1)

            # Select diverse samples based on diversity scores
            sorted_indices = np.argsort(diversity_scores)
            selected_indices = sorted_indices[:sample_size]
            cluster_samples = cluster_data[selected_indices]

        diverse_samples.append(cluster_samples)

    # Concatenate diverse samples from all clusters
    diverse_samples_array = np.concatenate(diverse_samples, axis=0)
    return diverse_samples_array


#####
# query-by-committee from StackGP
def ensembleSelect(models, inputData, responseData,
                   numberOfClusters=10):  # Generates a model ensemble using input data partitions
    data = np.transpose(inputData)
    if len(data) < numberOfClusters:
        numberOfClusters = len(data)
    clusters = KMeans(n_clusters=numberOfClusters).fit_predict(data)
    if numberOfClusters > len(set(clusters)):
        numberOfClusters = len(set(clusters))
        clusters = KMeans(n_clusters=numberOfClusters).fit_predict(data)
    dataParts = []
    partsResponse = []
    for i in range(numberOfClusters):
        dataParts.append([])
        partsResponse.append([])

    for i in range(len(clusters)):
        dataParts[clusters[i]].append(data[i])
        partsResponse[clusters[i]].append(responseData[i])

    modelResiduals = []

    for i in range(len(models)):
        modelResiduals.append([])
    for i in range(len(models)):
        for j in range(numberOfClusters):
            modelResiduals[i].append(fitness(models[i], np.transpose(dataParts[j]), partsResponse[j]))

    best = []
    for i in range(numberOfClusters):
        ordering = np.argsort(modelResiduals[i])
        j = 0
        while ordering[j] in best:
            j += 1
        best.append(ordering[j])

    ensemble = [models[best[i]] for i in range(numberOfClusters)]

    return ensemble


def uncertainty(data, trim=0.3):
    sortData = np.sort(data)
    trim = round(trim * len(data) / 2)
    if trim == 0:
        return np.std(data)
    return np.std(data[trim:-trim])


def evaluateModelEnsemble(ensemble, inputData):
    responses = [evaluateGPModel(mod, inputData) for mod in ensemble]
    if type(responses[0]) == np.ndarray:
        responses = np.transpose(responses)
        uncertainties = [uncertainty(res, 0.3) for res in responses]
        responses = [trim_mean(res, 0.3) for res in responses]
    else:

        uncertainties = [uncertainty(responses, 0.3)]
        responses = [trim_mean(responses, 0.3)]

    return responses, uncertainties


def relativeEnsembleUncertainty(ensemble, inputData):
    output = evaluateModelEnsemble(ensemble, inputData)
    return np.array(output[1]) / abs(np.array(output[0]))


def createUncertaintyFunc(ensemble):
    return lambda x: -relativeEnsembleUncertainty(ensemble, x)


def maximizeUncertainty(ensemble, varCount, bounds=[]):  # Used to select a new point of maximum uncertainty
    func = createUncertaintyFunc(ensemble)
    x0 = [1 for i in range(varCount)]
    if bounds == []:
        pt = minimize(func, x0).x
    else:
        pt = minimize(func, x0, bounds=bounds).x
    return pt


def extendData(data, newPoint):
    return np.concatenate((data.T, np.array([newPoint]))).T
