import numpy as np

def generate_complex_sample(n: int) -> tuple[np.ndarray, np.ndarray]:
    def f(x):
        return np.cos(np.sin(x[:,0]**2) * x[:,1]**2 + x[:,2]**3 + (4 * x[:,3])**2 + x[:,4]*10)
    X = np.random.randn(n, 5)
    y = f(X)
    return X.astype(np.float32), y.astype(np.float32)

def generate_simple_sample(n: int) -> tuple[np.ndarray, np.ndarray]:
    def f(x, y):
        return x**2 + y**2
    X = np.random.randn(n, 2)
    y = f(X[:, 0], X[:, 1])
    return X.astype(np.float32), y.astype(np.float32)
