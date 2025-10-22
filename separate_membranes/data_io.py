import numpy as np
import pandas as pd


def load_point_cloud(path: str) -> np.ndarray:
    """Return Nx3 array of (x,y,z) using columns mx,my,mz."""
    df = pd.read_csv(path)
    for col in ("mx", "my", "mz"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {path}")
    pts = df[["mx", "my", "mz"]].to_numpy(dtype=float)
    return pts


def standardise_points(points: np.ndarray):
    """Center and scale each axis; return (points_norm, means, stds)."""
    means = points.mean(axis=0)
    stds = points.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    norm = (points - means) / stds
    return norm, means, stds


def remove_outliers(points, k=16, density_quantile=0.02, mad_thresh=4.0):
    """Simple density + MAD filter on z."""
    from sklearn.neighbors import NearestNeighbors
    
    X = points[:, :2]
    z = points[:, 2]

    nbrs = NearestNeighbors(n_neighbors=min(k, len(points))).fit(X)
    dists, _ = nbrs.kneighbors(X)
    dens = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-9)
    keep_density = dens >= np.quantile(dens, density_quantile)

    med = np.median(z)
    mad = np.median(np.abs(z - med)) + 1e-12
    keep_mad = np.abs(z - med) <= mad_thresh * mad

    keep = keep_density & keep_mad
    return points[keep], keep
