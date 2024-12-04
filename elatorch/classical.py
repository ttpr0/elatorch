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

def nearest_better_clustering(
    X: torch.Tensor, y: torch.Tensor,
    minimize: bool = True
) -> dict:
    """
    Differentiable PyTorch implementation of Nearest Better Clustering (NBC) features.

    Parameters
    ----------
    X : torch.Tensor
        Tensor containing the decision space samples (n_samples x n_features).
    y : torch.Tensor
        Tensor containing the respective objective values (n_samples).
    minimize : bool, optional
        Whether the objective function is to be minimized, by default True.

    Returns
    -------
        Dictionary containing the calculated NBC features.
    """
    def compute_distances(X):
        # Compute pairwise squared distances
        diff = X[:, None, :] - X[None, :, :]
        sq_distances = (diff ** 2).sum(dim=-1)
        distances = torch.sqrt(sq_distances + 1e-12)  # Avoid zero for numerical stability
        return distances

    def nearest_neighbors(distances):
        # Get the indices of the nearest neighbor for each point (excluding itself)
        _, indices = distances.topk(2, dim=-1, largest=False)
        indices = indices[:, 1:] # Exclude self (distance = 0)
        return torch.gather(distances, 1, indices).flatten(), indices.flatten()

    def nearest_better_neighbors(X, y, distances):
        _distances = distances.clone()
        n_samples = X.size(0)
        nb_distances = torch.full((n_samples,), float('inf'), device=X.device)
        nb_indices = torch.full((n_samples,), -1, dtype=torch.long, device=X.device)
        for idx in range(n_samples):
            dist = _distances[idx, :]
            better = y < y[idx]
            if better.any():
                dist[~better] = float('inf')
                nb_idx = dist.argmin()
                nb_distances[idx] = distances[idx, nb_idx]
                nb_indices[idx] = nb_idx
            else:
                better = y == y[idx]
                better[idx] = False
                if better.any():
                    dist[~better] = float('inf')
                    nb_idx = dist.argmin()
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
    y = y if minimize else -y

    # Compute pairwise distances
    distances = compute_distances(X)
    # Find nearest neighbors
    nn_distances, nn_indices = nearest_neighbors(distances)
    # Find nearest better neighbors
    nb_distances, nb_indices = nearest_better_neighbors(X, y, distances)

    # Replace possible NA values (no better neighbour found) with distance to the nearest neighbour
    nb_no_nan = torch.isfinite(nb_distances)
    nb_distances[~nb_no_nan] = nn_distances[~nb_no_nan]
    nn_nb_ratio = torch.divide(nn_distances, nb_distances)
    nb_counts = torch.bincount(nb_indices[nb_indices >= 0], minlength=X.size(0)).to(torch.float32)

    nn_nb_sd_ratio = torch.std(nn_distances) / torch.std(nb_distances)
    nn_nb_mean_ratio = torch.mean(nn_distances) / torch.mean(nb_distances)
    nn_nb_correlation = compute_correlation(nn_distances, nb_distances)
    dist_ratio_coeff_var = torch.std(nn_nb_ratio) / torch.mean(nn_nb_ratio)
    nb_fitness_correlation = compute_correlation(nb_counts, y)

    return {
        'nbc.nn_nb.sd_ratio': nn_nb_sd_ratio,
        'nbc.nn_nb.mean_ratio': nn_nb_mean_ratio,
        'nbc.nn_nb.cor': nn_nb_correlation,
        'nbc.dist_ratio.coeff_var': dist_ratio_coeff_var,
        'nbc.nb_fitness.cor': nb_fitness_correlation,
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
