import math

import torch
from pflacco.misc_features import calculate_fitness_distance_correlation

from elatorch.misc import fitness_distance_correlation
from .util import generate_complex_sample, generate_simple_sample

def test_ela_meta_simple():
    X, y = generate_simple_sample(100)

    meta = calculate_fitness_distance_correlation(X, y)
    meta_torch = fitness_distance_correlation(torch.tensor(X), torch.tensor(y))

    for k, v in meta_torch.items():
        assert math.isclose(v, meta[k]), f"{k} does not match"

def test_ela_meta_complex():
    X, y = generate_complex_sample(100)

    meta = calculate_fitness_distance_correlation(X, y)
    meta_torch = fitness_distance_correlation(torch.tensor(X), torch.tensor(y))

    for k, v in meta_torch.items():
        assert math.isclose(v, meta[k]), f"{k} does not match"
