from typing import Callable
import math

import torch

def _neff(n: int, weights: torch.Tensor | None = None) -> float:
    if weights is None:
        weights = torch.ones((n,))/ n
    return 1 / torch.sum(weights**2)

def scotts_factor(n: int, d: int) -> float:
    """Compute Scott's factor.
    """
    return torch.pow(_neff(n), -1./(d+4))

def silverman_factor(n: int, d: int) -> float:
    """Compute the Silverman factor.
    """
    return torch.pow(_neff(n)*(d+2.0)/4.0, -1./(d+4))

def gaussian_kde(data: torch.Tensor, points: torch.Tensor, covariance_factor: Callable[[int, int], float] = scotts_factor):
    """

    Parameters
    ----------
    data : torch.Tensor
        [N x D] Tensor containing the N data points with D dimensions.

    """
    data = torch.atleast_2d(data).T
    n, d = data.shape
    if d > n:
        msg = ("Number of dimensions is greater than number of samples. "
                "This results in a singular data covariance matrix, which "
                "cannot be treated using the algorithms implemented in "
                "`gaussian_kde`. Note that `gaussian_kde` interprets each "
                "*column* of `dataset` to be a point; consider transposing "
                "the input to `dataset`.")
        raise ValueError(msg)

    factor = covariance_factor(n, d)
    weights = torch.ones((n,1))/ n

    _data_covariance = torch.atleast_2d(torch.cov(data.T))
    _data_cho_cov = torch.linalg.cholesky(_data_covariance)
    covariance = _data_covariance * factor**2
    cho_cov = (_data_cho_cov * factor).to(torch.float32)
    log_det = 2*torch.log(torch.diag(cho_cov * math.sqrt(2*math.pi))).sum()

    points = torch.atleast_2d(points).T
    if points.shape[1] != d:
        raise ValueError("points and xi must have same dimension")
    m, _ = points.shape

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
