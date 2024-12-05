from typing import Callable
import math

import torch

def _neff(weights: torch.Tensor) -> torch.Tensor:
    return 1 / torch.sum(weights**2)

def scotts_factor(n: int, d: int, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Compute Scott's factor.
    """
    if weights is None:
        weights = torch.ones((n,1), dtype=torch.float32)/ n
    return torch.pow(_neff(weights), -1./(d+4))

def silverman_factor(n: int, d: int, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the Silverman factor.
    """
    if weights is None:
        weights = torch.ones((n,1), dtype=torch.float32)/ n
    return torch.pow(_neff(weights)*(d+2.0)/4.0, -1./(d+4))

def gaussian_kde(data: torch.Tensor, points: torch.Tensor, covariance_factor = scotts_factor) -> torch.Tensor:
    """
    Gaussian Kernel Density Estimation.

    Parameters
    ----------
    data : torch.Tensor
        [N x D] Tensor containing the N data points with D dimensions.
    points : torch.Tensor
        [M x D] Tensor containing the M query points with D dimensions.
    covariance_factor : Callable, optional
        The covariance factor function, by default scotts_factor

    Returns
    -------
    torch.Tensor
        [M] Tensor containing the estimated density values at the query points.
    """
    if data.device != points.device:
        raise ValueError("Data and points must be on the same device.")
    if data.dtype != torch.float32 or points.dtype != torch.float32:
        raise ValueError("Data and points must be of type torch.float32.")
    data = torch.atleast_2d(data).T
    n, d = data.shape
    if d > n:
        raise ValueError("The number of data points must be greater than the number of dimensions.")

    weights = torch.ones((n,1), dtype=torch.float32, device=data.device)/ n
    factor = covariance_factor(n, d, weights)

    _data_covariance = torch.atleast_2d(torch.cov(data.T))
    _data_cho_cov = torch.linalg.cholesky(_data_covariance)
    cho_cov = (_data_cho_cov * factor).to(torch.float32)

    points = torch.atleast_2d(points).T
    if points.shape[1] != d:
        raise ValueError("points and xi must have same dimension")

    # Rescale the data
    data_ = torch.linalg.solve_triangular(cho_cov, data.T, upper=False).T
    points_ = torch.linalg.solve_triangular(cho_cov, points.T, upper=False).T

    # Evaluate the normalisation
    norm = math.pow((2 * math.pi), (- d / 2.))
    norm = norm / torch.trace(cho_cov).prod()

    diff = data_[:, None, :] - points_[None, :, :]
    sq_diff = (diff ** 2).sum(dim=-1)
    arg = torch.exp(-sq_diff / 2.) * norm # squard distances between points [n x m]
    estimate = (weights.T @ arg).T

    return estimate[:, 0]
