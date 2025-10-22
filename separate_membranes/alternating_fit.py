import numpy as np
from .design_matrix import cubic_design_matrix
from .qp_solver import fit_two_surfaces_with_gap
from .data_io import standardise_points, remove_outliers


def _midplane_init(x, y, z, alpha=1e-3):
    """Initialize labels using a simple midplane fit."""
    A = cubic_design_matrix(x, y)
    ATA = A.T @ A + alpha * np.eye(A.shape[1])
    coeff = np.linalg.solve(ATA, A.T @ z)
    g = A @ coeff
    labels = (z >= g).astype(int)  # 1=top/upper, 0=bottom/lower
    return labels


def alternating_surface_fit(x, y, z, Delta=None, alpha=1e-3,
                            max_iter=10, tol=0.01, soft_gap=False, lambda_gap=10.0,
                            flat_surfaces=False):
    """Fit two membrane surfaces using alternating optimization.
    
    Args:
        x (np.ndarray): The x-coordinates of the points.
        y (np.ndarray): The y-coordinates of the points.
        z (np.ndarray): The z-coordinates of the points.
        Delta (float): The gap size. Defaults to None, in which case it is computed from the data.
        alpha (float): The regularisation parameter for ridge regression. Used to control the trade-off between fitting the data and enforcing smoothness during initial surface fitting. Defaults to 1e-3.
        max_iter (int): The maximum number of iterations. Defaults to 10.
        tol (float): The tolerance for the convergence. Defaults to 0.01.
        soft_gap (bool): Whether to use a soft gap constraint. If false, the surfaces will always be separated by a fixed gap size (Delta), else a soft gap constraint is used to allow the gap size to vary. Defaults to False.
        lambda_gap (float): The weight for the soft gap constraint. Only used if soft_gap is True. Defaults to 10.0.
        flat_surfaces (bool): Whether to try enforcing flat surfaces. Defaults to False.
    """
    pts = np.column_stack([x, y, z])
    pts_filt, keep_mask = remove_outliers(pts)
    xF, yF, zF = pts_filt[:,0], pts_filt[:,1], pts_filt[:,2]

    ptsN, means, stds = standardise_points(pts_filt)
    xN, yN, zN = ptsN[:,0], ptsN[:,1], ptsN[:,2]

    if Delta is None:
        zq = np.quantile(zN, [0.05, 0.95])
        Delta = 0.3 * (zq[1] - zq[0])

    labels = _midplane_init(xN, yN, zN, alpha=alpha)

    A = cubic_design_matrix(xN, yN)

    for it in range(max_iter):
        θt, θb = fit_two_surfaces_with_gap(
            A, zN, labels, Delta=Delta, alpha=alpha,
            soft_gap=soft_gap, lambda_gap=lambda_gap,
            flat_surfaces=flat_surfaces
        )
        
        if θt is None or θb is None:
            print(f"Warning: QP solver failed at iteration {it+1}. Using previous solution or fallback.")
            if it == 0:
                print("Using simple midplane fit as fallback.")
                ATA = A.T @ A + alpha * np.eye(A.shape[1])
                coeff = np.linalg.solve(ATA, A.T @ zN)
                θt = coeff + 0.1 * np.random.randn(len(coeff))
                θb = coeff - 0.1 * np.random.randn(len(coeff))
            else:
                break
        
        zt_pred = A @ θt
        zb_pred = A @ θb
        rt = np.abs(zN - zt_pred)
        rb = np.abs(zN - zb_pred)
        new_labels = (rt <= rb).astype(int)

        change = np.mean(new_labels != labels)
        labels = new_labels
        if change < tol:
            break

    model = {
        "theta_top": θt.tolist(),
        "theta_bottom": θb.tolist(),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "Delta_norm": float(Delta),
        "iterations": it + 1
    }
    
    rmse_t = float(np.sqrt(np.mean((zN[labels==1] - zt_pred[labels==1])**2))) if np.any(labels==1) else float("nan")
    rmse_b = float(np.sqrt(np.mean((zN[labels==0] - zb_pred[labels==0])**2))) if np.any(labels==0) else float("nan")
    min_gap = float(np.min(zt_pred - zb_pred))
    metrics = {"RMSE_top_norm": rmse_t, "RMSE_bottom_norm": rmse_b, "achieved_min_gap_norm": min_gap}

    full_labels = np.zeros(len(pts), dtype=int)
    full_labels[:] = -1
    full_labels[np.where(keep_mask)[0]] = labels

    return {
        "labels_filtered": labels,
        "labels_full": full_labels,  # -1 for filtered-out points
        "filtered_points": pts_filt,
        "model": model,
        "metrics": metrics
    }
