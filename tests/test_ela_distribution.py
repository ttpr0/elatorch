import math

import torch
from pflacco.classical_ela_features import calculate_ela_distribution

from elatorch.classical import ela_distribution
from .util import generate_complex_sample, generate_simple_sample

def test_ela_distribution_simple():
    X, y = generate_simple_sample(100)

    meta = calculate_ela_distribution(X, y)
    meta_torch = ela_distribution(torch.tensor(X), torch.tensor(y))

    for k, v in meta_torch.items():
        assert math.isclose(v, meta[k]), f"{k} does not match"

def test_ela_distribution_complex():
    X, y = generate_complex_sample(100)

    meta = calculate_ela_distribution(X, y)
    meta_torch = ela_distribution(torch.tensor(X), torch.tensor(y))

    for k, v in meta_torch.items():
        assert math.isclose(v, meta[k]), f"{k} does not match"
