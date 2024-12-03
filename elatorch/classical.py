import math

import torch

from .utils.kde import gaussian_kde, scotts_factor

def ela_meta(X: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    PyTorch differentiable implementation of the ELA meta features calculation.

    Parameters
    ----------
    X: torch.Tensor
        A tensor containing a sample of the decision space.
    y: torch.Tensor
        A tensor containing the respective objective values of `X`.

    Returns
    -------
        Dictionary consisting of the calculated features.
    """
    def adjusted_r2(y_true, y_pred, n_params):
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        n = y_true.size(0)
        return 1 - (1 - r2) * (n - 1) / (n - n_params - 1)

    def fit_linear_model(X, y):
        # Add bias term to X
        X_with_bias = torch.cat([torch.ones((X.size(0), 1)), X], dim=1)
        # Compute weights using the closed-form solution
        weights = torch.linalg.lstsq(X_with_bias, y).solution
        bias = weights[0]
        coefs = weights[1:]
        return bias, coefs

    def predict_linear_model(X, bias, coefs):
        return X @ coefs + bias

    # Fit simple linear model
    lin_simple_bias, lin_simple_coefs = fit_linear_model(X, y)
    lin_simple_pred = predict_linear_model(X, lin_simple_bias, lin_simple_coefs)
    lin_simple_intercept = lin_simple_bias
    lin_simple_coef_min = lin_simple_coefs.abs().min()
    lin_simple_coef_max = lin_simple_coefs.abs().max()
    lin_simple_coef_max_by_min = lin_simple_coef_max / lin_simple_coef_min
    lin_simple_adj_r2 = adjusted_r2(y, lin_simple_pred, X.size(1))

    # Fit linear model with interactions
    interaction_terms = []
    for i in range(X.size(1)):
        for j in range(i + 1, X.size(1)):
            interaction_terms.append((X[:, i] * X[:, j]).reshape((-1, 1)))
    if interaction_terms:
        X_interact = torch.cat([X] + interaction_terms, dim=1)
    else:
        X_interact = X
    lin_w_interact_bias, lin_w_interact_coefs = fit_linear_model(X_interact, y)
    lin_w_interact_pred = predict_linear_model(X_interact, lin_w_interact_bias, lin_w_interact_coefs)
    lin_w_interact_adj_r2 = adjusted_r2(y, lin_w_interact_pred, X_interact.size(1))

    # Fit quadratic model
    X_squared = torch.cat([X, X.pow(2)], dim=1)
    quad_simple_bias, quad_simple_coefs = fit_linear_model(X_squared, y)
    quad_simple_pred = predict_linear_model(X_squared, quad_simple_bias, quad_simple_coefs)
    quad_simple_adj_r2 = adjusted_r2(y, quad_simple_pred, X_squared.size(1))
    quad_model_coefs = quad_simple_coefs[X.size(1):]
    quad_model_con_min = quad_model_coefs.abs().min()
    quad_model_con_max = quad_model_coefs.abs().max()
    quad_simple_cond = quad_model_con_max / quad_model_con_min

    # Fit quadratic model with interactions
    interaction_quad_terms = []
    for i in range(X_squared.size(1)):
        for j in range(i + 1, X_squared.size(1)):
            interaction_quad_terms.append((X_squared[:, i] * X_squared[:, j]).reshape((-1, 1)))
    if interaction_quad_terms:
        X_quad_interact = torch.cat([X_squared] + interaction_quad_terms, dim=1)
    else:
        X_quad_interact = X_squared
    quad_w_interact_bias, quad_w_interact_coefs = fit_linear_model(X_quad_interact, y)
    quad_w_interact_pred = predict_linear_model(X_quad_interact, quad_w_interact_bias, quad_w_interact_coefs)
    quad_w_interact_adj_r2 = adjusted_r2(y, quad_w_interact_pred, X_quad_interact.size(1))

    return {
        'ela_meta.lin_simple.adj_r2': lin_simple_adj_r2,
        'ela_meta.lin_simple.intercept': lin_simple_intercept,
        'ela_meta.lin_simple.coef.min': lin_simple_coef_min,
        'ela_meta.lin_simple.coef.max': lin_simple_coef_max,
        'ela_meta.lin_simple.coef.max_by_min': lin_simple_coef_max_by_min,
        'ela_meta.lin_w_interact.adj_r2': lin_w_interact_adj_r2,
        'ela_meta.quad_simple.adj_r2': quad_simple_adj_r2,
        'ela_meta.quad_simple.cond': quad_simple_cond,
        'ela_meta.quad_w_interact.adj_r2': quad_w_interact_adj_r2,
    }

def nearest_better_clustering_2(
    X: torch.Tensor, y: torch.Tensor, fast_k: float = 0.05,
    dist_tie_breaker: str = 'sample', minimize: bool = True
) -> dict:
    """
    Differentiable PyTorch implementation of Nearest Better Clustering (NBC) features.

    Parameters
    ----------
    X : torch.Tensor
        Tensor containing the decision space samples (n_samples x n_features).
    y : torch.Tensor
        Tensor containing the respective objective values (n_samples).
    fast_k : float, optional
        Percentage of observations to consider when searching for nearest better neighbors, by default 0.05.
    dist_tie_breaker : str, optional
        Strategy for breaking ties ('sample', 'first', 'last'), by default 'sample'.
    minimize : bool, optional
        Whether the objective function is to be minimized, by default True.

    Returns
    -------
        Dictionary containing the calculated NBC features.
    """
    def adjusted_objective(y, minimize):
        return y if minimize else -y

    def compute_distances(X):
        # Compute pairwise squared distances
        diff = X[:, None, :] - X[None, :, :]
        sq_distances = (diff ** 2).sum(dim=-1)
        distances = torch.sqrt(sq_distances + 1e-12)  # Avoid zero for numerical stability
        return distances

    def find_nearest_neighbors(distances, k):
        # Get the indices of the k nearest neighbors for each point (excluding itself)
        _, indices = distances.topk(k + 1, dim=-1, largest=False)
        return indices[:, 1:]  # Exclude self (distance = 0)

    def nearest_better_neighbors(X, y, distances):
        n_samples = X.size(0)
        nb_distances = torch.full((n_samples,), float('inf'), device=X.device)
        nb_indices = torch.full((n_samples,), -1, dtype=torch.long, device=X.device)
        for idx in range(n_samples):
            dist = distances[idx, :].clone()
            better = y < y[idx]
            if better.any():
                dist[better] = float('inf')
                nb_idx = distances.argmin()
                nb_distances[idx] = distances[idx, nb_idx]
                nb_indices[idx] = nb_idx
        return nb_distances, nb_indices

    def compute_correlation(x, y):
        # Calculate Pearson correlation coefficient
        x_mean, y_mean = x.mean(), y.mean()
        cov = ((x - x_mean) * (y - y_mean)).mean()
        std_x = torch.sqrt(((x - x_mean) ** 2).mean())
        std_y = torch.sqrt(((y - y_mean) ** 2).mean())
        return cov / (std_x * std_y + 1e-12)  # Avoid division by zero

    # Adjust objective values
    y = adjusted_objective(y, minimize)

    # Number of samples to consider in fast_k
    k = max(1, int(fast_k * X.size(0)))

    # Compute pairwise distances
    distances = compute_distances(X)

    # Find nearest neighbors
    nn_indices = find_nearest_neighbors(distances, k)

    # Find nearest better neighbors
    nb_distances, nb_indices = nearest_better_neighbors(X, y, distances)

    # Replace NaNs in nb_distances with nearest neighbor distances
    nb_distances[torch.isinf(nb_distances)] = distances[torch.arange(X.size(0)), nn_indices[:, 0]][torch.isinf(nb_distances)]

    # Calculate nn_nb features
    nn_distances = distances[torch.arange(X.size(0)), nn_indices[:, 0]]
    nn_nb_sd_ratio = nn_distances.std(unbiased=True) / nb_distances.std(unbiased=True)
    nn_nb_mean_ratio = nn_distances.mean() / nb_distances.mean()
    nn_nb_correlation = compute_correlation(nn_distances, nb_distances)

    # Calculate dist_ratio coefficient of variation
    dist_ratio = nn_distances / nb_distances
    dist_ratio_coeff_var = dist_ratio.std(unbiased=True) / dist_ratio.mean()

    # Calculate nb_fitness correlation
    indegree = torch.zeros_like(y)
    for idx in range(nb_indices.size(0)):
        if nb_indices[idx] != -1:
            indegree[nb_indices[idx]] += 1
    nb_fitness_correlation = compute_correlation(indegree, y)

    return {
        'nbc.nn_nb.sd_ratio': nn_nb_sd_ratio,
        'nbc.nn_nb.mean_ratio': nn_nb_mean_ratio,
        'nbc.nn_nb.cor': nn_nb_correlation,
        'nbc.dist_ratio.coeff_var': dist_ratio_coeff_var,
        'nbc.nb_fitness.cor': nb_fitness_correlation,
    }

def nearest_better_clustering(
    X: torch.Tensor,  # Changed to torch.Tensor
    y: torch.Tensor,  # Changed to torch.Tensor
    fast_k: float = 0.05,
    dist_tie_breaker: str = 'sample',
    minimize: bool = True
) -> dict[str, int | float]:
    """
    Differentiable PyTorch implementation of Nearest Better Clustering (NBC) features.

    Parameters
    ----------
    X : torch.Tensor
        Tensor containing the decision space samples (n_samples x n_features).
    y : torch.Tensor
        Tensor containing the respective objective values (n_samples).
    fast_k : float, optional
        Percentage of observations to consider when searching for nearest better neighbors, by default 0.05.
    dist_tie_breaker : str, optional
        Strategy for breaking ties ('sample', 'first', 'last'), by default 'sample'.
    minimize : bool, optional
        Whether the objective function is to be minimized, by default True.

    Returns
    -------
        Dictionary containing the calculated NBC features.
    """
    # Ensure inputs are tensors
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    # Ensure that y is a column vector
    y = y.view(-1)

    # Adjust fast_k to be an integer
    if fast_k < 1:
        fast_k = math.ceil(fast_k * X.shape[0])
    if fast_k < 0 or fast_k > X.shape[0]:
        raise ValueError(f'[{fast_k}] of "fast_k" does not lie in the interval [0,n] where n is the number of observations.')
    if not minimize:
        y = y * -1

    # Compute pairwise distances (Euclidean distance)
    def compute_distances(X):
        diff = X.unsqueeze(1) - X.unsqueeze(0)  # Shape (n, n, d)
        return torch.norm(diff, dim=2)  # Shape (n, n)

    dist_matrix = compute_distances(X)

    # Find the nearest neighbors (ignoring the self distance)
    _, indices = torch.topk(dist_matrix, fast_k + 1, largest=False, sorted=False)

    results = []

    for idx in range(X.shape[0]):
        # for every sample 'idx' find the BETTER neighbours 'nnb' out of the 'fast_k' nearest neighbours
        y_rec = y[idx]
        ind_nn = indices[idx, 1:]  # Exclude self
        y_near = y[ind_nn]
        better = (y_near < y_rec).to(torch.int32)

        # If there are better neighbors, select the closest one
        if better.sum() > 0:
            b_idx = better.argmax()
            results.append([idx, ind_nn[b_idx], dist_matrix[idx, ind_nn[b_idx + 1]]])

        # If no better neighbors, get the nearest better neighbor from the entire dataset
        else:
            # Get the indices of all other points
            # ind_alt = torch.setdiff1d(torch.arange(X.shape[0]), indices[idx])
            combined = torch.cat((indices[idx], torch.arange(X.shape[0])))
            uniques, counts = combined.unique(return_counts=True)
            ind_alt = uniques[counts == 1]
            alt_y = y[ind_alt]

            # Find valid alternatives (better neighbors)
            valid_alt = ind_alt[alt_y < y_rec]
            if valid_alt.size(0) == 0:
                valid_alt = ind_alt[alt_y == y_rec]
            if valid_alt.size(0) == 0:
                results.append([idx, torch.tensor(torch.nan), torch.tensor(torch.nan)])
                continue

            # Compute distances for alternatives
            alt_dist = torch.norm(X[valid_alt] - X[idx], dim=1)
            min_dist_idx = alt_dist.argmin()

            if len(min_dist_idx) > 1:
                if dist_tie_breaker == 'sample':
                    min_dist_idx = min_dist_idx[torch.randint(len(min_dist_idx), (1,))]
                elif dist_tie_breaker == 'first':
                    min_dist_idx = min_dist_idx[0]
                elif dist_tie_breaker == 'last':
                    min_dist_idx = min_dist_idx[-1]
                else:
                    raise ValueError('Possible tie breaker methods are "sample", "first", and "last"')

            results.append([idx, valid_alt[min_dist_idx], alt_dist[min_dist_idx]])

    nb_stats = torch.tensor(results, dtype=torch.float32)
    near_dist = dist_matrix[:, 1:].mean(dim=1)  # Average nearest distances
    near_better_dist = nb_stats[:, 2]
    nb_near_ratio = near_better_dist / near_dist

    nb_stats = torch.cat((nb_stats, near_dist.view(-1, 1), nb_near_ratio.view(-1, 1), y.view(-1, 1)), dim=1)

    # Compute fitness statistics
    result_stats = []
    for own_id in range(X.shape[0]):
        x = nb_stats[:, 1] == own_id
        count = x.sum()
        if count > 0:
            to_me_dist = near_better_dist[x].nanmedian()
            result_stats.append([count, to_me_dist, near_better_dist[own_id] / to_me_dist])
        else:
            result_stats.append([0, torch.tensor(torch.nan), torch.tensor(torch.nan)])

    result_stats = torch.tensor(result_stats)

    # Replace possible NA values (occurring when no better nearer neighbour is found) with the distance to the closest nearest neighbour
    near_better_dist = torch.where(torch.isnan(near_better_dist), near_dist, near_better_dist)
    dist_ratio = near_dist / near_better_dist

    # Calculate Pearson correlation using PyTorch
    def pearson_corr(x, y):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        std_x = torch.std(x)
        std_y = torch.std(y)
        return torch.mean((x - mean_x) * (y - mean_y)) / (std_x * std_y)

    return {
        'nbc.nn_nb.sd_ratio': torch.std(near_dist) / torch.std(near_better_dist),
        'nbc.nn_nb.mean_ratio': torch.mean(near_dist) / torch.mean(near_better_dist),
        'nbc.nn_nb.cor': pearson_corr(near_dist, near_better_dist),
        'nbc.dist_ratio.coeff_var': torch.std(dist_ratio) / torch.mean(dist_ratio),
        'nbc.nb_fitness.cor': pearson_corr(result_stats[:, 0], y)
    }

def ela_distribution(
    X: torch.Tensor,
    y: torch.Tensor,
    ela_distr_skewness_type: int = 3,
    ela_distr_kurtosis_type: int = 3
) -> dict[str, int | float]:
    """ELA Distribution features using PyTorch for differentiable computation.

    Parameters
    ----------
    X : torch.Tensor
        Tensor containing decision space samples (n_samples x n_features).
    y : torch.Tensor
        Tensor containing objective values (n_samples).
    ela_distr_skewness_type : int, optional
        Skewness type, by default 3.
    ela_distr_kurtosis_type : int, optional
        Kurtosis type, by default 3.

    Returns
    -------
    dict[str, int | float]
        Dictionary containing ELA Distribution features.
    """
    if ela_distr_skewness_type not in range(1,4):
        raise Exception('Skewness type must be an integer and in the intervall [1,3]')
    if ela_distr_kurtosis_type not in range(1,4):
        raise Exception('Kurtosis type must be an integer and in the intervall [1,3]')

    # Remove NaN values (PyTorch doesn't have a nan mask, so use finite checks)
    y = y[torch.isfinite(y)]
    n = y.shape[0]

    if n < 4:
        raise ValueError('At least 4 complete observations are required')

    # Calculate skewness
    y_mean = y.mean()
    y_centered = y - y_mean
    m2 = (y_centered.pow(2)).sum()
    m3 = (y_centered.pow(3)).sum()
    skewness = torch.sqrt(torch.tensor(n, dtype=torch.float32)) * m3 / (m2 ** 1.5)
    if ela_distr_skewness_type == 2:
        skewness = skewness * torch.sqrt(n * (n - 1)) / (n - 2)
    elif ela_distr_skewness_type == 3:
        skewness = skewness * (1 - 1/n) ** 1.5

    # Calculate kurtosis
    m4 = (y_centered.pow(4)).sum()
    kurtosis = n * m4 / (m2 ** 2)
    if ela_distr_kurtosis_type == 1:
        kurtosis = kurtosis - 3
    elif ela_distr_kurtosis_type == 2:
        kurtosis = ((n + 1) * (kurtosis - 3) + 6) * (n - 1) / ((n - 2) * (n - 3))
    elif ela_distr_kurtosis_type == 3:
        kurtosis = kurtosis * ((1 - 1/n) ** 2) - 3

    # Estimate the number of peaks using a Gaussian kernel density approximation
    y_ = torch.atleast_2d(y).T
    n, d = y_.shape
    covariance_factor = scotts_factor(n, d)
    low = y.min() - 3 * covariance_factor * y.std()
    high = y.max() + 3 * covariance_factor * y.std()
    m = 512
    positions = torch.linspace(low, high, m)
    estimate = gaussian_kde(y, positions)
    grad = torch.diff(estimate)
    peaks = torch.logical_and(grad[:-1] < 0, grad[1:] > 0)
    # n_peaks = torch.sum(peaks)
    peaks = torch.cat([torch.tensor([True]), peaks, torch.tensor([True])])
    indices = torch.nonzero(peaks)
    modemass = torch.zeros((indices.size(0),))
    for idx in range(indices.size(0) - 1):
        a = indices[idx]
        b = indices[idx + 1] - 1
        modemass[idx] = estimate[a:b].mean() + torch.abs(positions[a] - positions[b])
    n_peaks = (modemass > 0.1).sum()

    return {
        'ela_distr.skewness': skewness,
        'ela_distr.kurtosis': kurtosis,
        'ela_distr.number_of_peaks': n_peaks
    }
