import math

import torch
from pflacco.classical_ela_features import calculate_nbc

from elatorch.classical import nearest_better_clustering
from .util import generate_complex_sample, generate_simple_sample

def test_nbc_simple():
    X, y = generate_simple_sample(100)

    meta = calculate_nbc(X, y)
    meta_torch = nearest_better_clustering(torch.tensor(X), torch.tensor(y))

    for k, v in meta_torch.items():
        assert math.isclose(v, meta[k]), f"{k} does not match"

def test_nbc_complex():
    X, y = generate_complex_sample(100)

    meta = calculate_nbc(X, y)
    meta_torch = nearest_better_clustering(torch.tensor(X), torch.tensor(y))

    for k, v in meta_torch.items():
        assert math.isclose(v, meta[k]), f"{k} does not match"
