import math

import torch
from pflacco.classical_ela_features import calculate_ela_meta

from elatorch.classical import ela_meta
from .util import generate_complex_sample, generate_simple_sample

def test_ela_meta_simple():
    X, y = generate_simple_sample(100)

    meta = calculate_ela_meta(X, y)
    meta_torch = ela_meta(torch.tensor(X), torch.tensor(y))

    for k, v in meta_torch.items():
        assert math.isclose(v, meta[k], rel_tol=1e-5), f"{k} does not match"

def test_ela_meta_complex():
    X, y = generate_complex_sample(100)

    meta = calculate_ela_meta(X, y)
    meta_torch = ela_meta(torch.tensor(X), torch.tensor(y))

    for k, v in meta_torch.items():
        assert math.isclose(v, meta[k], rel_tol=1e-5), f"{k} does not match"
