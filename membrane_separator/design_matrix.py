import numpy as np


def cubic_design_matrix(x, y) -> np.ndarray:
    """Nx10: [1, x, y, x², xy, y², x³, x²y, xy², y³]."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    return np.column_stack([
        np.ones_like(x),
        x, y,
        x*x, x*y, y*y,
        x*x*x, (x*x)*y, x*(y*y), y*y*y
    ])
