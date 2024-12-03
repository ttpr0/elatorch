import math

import torch

def fitness_distance_correlation(
    X: torch.Tensor, y: torch.Tensor, f_opt: float | None = None,
    proportion_of_best: float = 0.1, minimize: bool = True, minkowski_p: int = 2
) -> dict[str, int | float]:
    """
    Differentiable PyTorch implementation of fitness-distance correlation.

    Parameters
    ----------
    X : torch.Tensor
        Tensor containing decision space samples (n_samples x n_features).
    y : torch.Tensor
        Tensor containing objective values (n_samples).
    f_opt : float | None, optional
        Optimal fitness value to compare against, by default None.
    proportion_of_best : float, optional
        Proportion of the best samples to consider, by default 0.1.
    minimize : bool, optional
        Whether the objective is to be minimized, by default True.
    minkowski_p : int, optional
        Order of the Minkowski distance, by default 2.

    Returns
    -------
        Dictionary containing fitness-distance correlation and related features.
    """
    if proportion_of_best > 1 or proportion_of_best <= 0:
        raise ValueError("Proportion of the best samples must be in the interval (0, 1].")

    if not minimize:
        y = -y
    if f_opt is not None and not minimize:
        f_opt = -f_opt

    # Select the proportion of the best samples
    n_samples = X.size(0)
    n_best = max(2, int(proportion_of_best * n_samples))

    y_sorted, indices = torch.sort(y, descending=False)
    best_indices = indices[:n_best]
    X_best = X[best_indices]
    y_best = y_sorted[:n_best]

    # Determine index of f_opt or the minimum value
    if f_opt is None:
        fopt_idx = torch.argmin(y_best)
    else:
        fopt_idx = (y_best == f_opt).nonzero(as_tuple=True)[0]
        if len(fopt_idx) == 0:
            fopt_idx = torch.argmin(y_best)
        else:
            fopt_idx = fopt_idx[0]

    # Compute pairwise Minkowski distances
    def minkowski_distance(X1, X2, p):
        diff = X1[:, None, :] - X2[None, :, :]
        dist = torch.sum(torch.abs(diff) ** p, dim=-1) ** (1 / p)
        return dist

    distances = minkowski_distance(X_best, X_best, p=minkowski_p)
    dist_to_fopt = distances[fopt_idx]

    # Compute means and standard deviations
    dist_mean = dist_to_fopt.mean()
    dist_std = dist_to_fopt.std(unbiased=True)
    y_mean = y_best.mean()
    y_std = y_best.std(unbiased=True)

    # Compute covariance between fitness and distances
    fitness_diff = y_best - y_mean
    distance_diff = dist_to_fopt - dist_mean
    covariance_fd = torch.mean(fitness_diff * distance_diff)

    # Compute fitness-distance correlation
    correlation_fd = covariance_fd / (y_std * dist_std + 1e-12)  # Avoid division by zero

    return {
        "fitness_distance.fd_correlation": correlation_fd.item(),
        "fitness_distance.fd_cov": covariance_fd.item(),
        "fitness_distance.distance_mean": dist_mean.item(),
        "fitness_distance.distance_std": dist_std.item(),
        "fitness_distance.fitness_mean": y_mean.item(),
        "fitness_distance.fitness_std": y_std.item(),
    }
