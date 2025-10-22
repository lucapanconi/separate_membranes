import cvxpy as cp
import numpy as np
from .design_matrix import cubic_design_matrix


def fit_two_surfaces_with_gap(A, z, labels, Delta=0.3, alpha=1e-3,
                              soft_gap=False, lambda_gap=10.0,
                              solver="OSQP", flat_surfaces=False):
    """Fit two cubic surfaces with gap constraint using quadratic programming."""
    labels = np.asarray(labels).astype(bool)  # True=top/upper, False=bottom/lower
    idx_t = labels
    idx_b = ~labels
    At, Ab = A[idx_t], A[idx_b]
    zt, zb = z[idx_t], z[idx_b]

    p = A.shape[1]
    theta_t = cp.Variable(p)
    theta_b = cp.Variable(p)

    if flat_surfaces:
        reg_weights = np.ones(p)
        reg_weights[6:10] = 10.0
        reg_weights[3:6] = 3.0
        
        obj = cp.sum_squares(At @ theta_t - zt) + alpha * cp.sum_squares(cp.multiply(reg_weights, theta_t)) \
            + cp.sum_squares(Ab @ theta_b - zb) + alpha * cp.sum_squares(cp.multiply(reg_weights, theta_b))
    else:
        obj = cp.sum_squares(At @ theta_t - zt) + alpha * cp.sum_squares(theta_t) \
            + cp.sum_squares(Ab @ theta_b - zb) + alpha * cp.sum_squares(theta_b)

    constraints = []
    if soft_gap:
        gap_shortfall = cp.pos(Delta - (A @ theta_t - A @ theta_b))
        obj += lambda_gap * cp.sum_squares(gap_shortfall)
    else:
        constraints.append(A @ theta_t - A @ theta_b >= Delta)
    
    if flat_surfaces:
        x_min, x_max = np.min(A[:, 1]), np.max(A[:, 1])
        y_min, y_max = np.min(A[:, 2]), np.max(A[:, 2])
        
        x_margin = 0.1 * (x_max - x_min)
        y_margin = 0.1 * (y_max - y_min)
        
        boundary_x = np.concatenate([
            np.linspace(x_min - x_margin, x_max + x_margin, 10),
            np.linspace(x_min - x_margin, x_max + x_margin, 10),
            np.full(10, x_min - x_margin),
            np.full(10, x_max + x_margin)
        ])
        boundary_y = np.concatenate([
            np.full(10, y_min - y_margin),
            np.full(10, y_max + y_margin),
            np.linspace(y_min - y_margin, y_max + y_margin, 10),
            np.linspace(y_min - y_margin, y_max + y_margin, 10)
        ])
        
        A_boundary = cubic_design_matrix(boundary_x, boundary_y)
        
        z_mean = np.mean(np.concatenate([zt, zb]))
        max_deviation = 0.5 * np.std(np.concatenate([zt, zb]))
        
        boundary_penalty = 1.0
        obj += boundary_penalty * cp.sum_squares(cp.pos(cp.abs(A_boundary @ theta_t - z_mean) - max_deviation))
        obj += boundary_penalty * cp.sum_squares(cp.pos(cp.abs(A_boundary @ theta_b - z_mean) - max_deviation))

    prob = cp.Problem(cp.Minimize(obj), constraints)
    
    solvers_to_try = [
        ("OSQP", {"eps_abs": 1e-4, "eps_rel": 1e-4, "max_iter": 10000, "verbose": False}),
        ("OSQP", {"eps_abs": 1e-3, "eps_rel": 1e-3, "max_iter": 5000, "verbose": False}),
        ("ECOS", {"max_iters": 1000, "verbose": False}),
        ("SCS", {"max_iters": 5000, "verbose": False})
    ]
    
    last_error = None
    for solver_name, solver_params in solvers_to_try:
        try:
            if solver_name == "OSQP":
                prob.solve(solver=cp.OSQP, **solver_params)
            elif solver_name == "ECOS":
                prob.solve(solver=cp.ECOS, **solver_params)
            elif solver_name == "SCS":
                prob.solve(solver=cp.SCS, **solver_params)
            
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return theta_t.value, theta_b.value
            else:
                last_error = f"Solver {solver_name} failed with status: {prob.status}"
                
        except Exception as e:
            last_error = f"Solver {solver_name} failed with exception: {str(e)}"
            continue
    
    if not soft_gap:
        print(f"Warning: Hard gap constraint failed. Trying soft gap constraint as fallback.")
        obj_soft = cp.sum_squares(At @ theta_t - zt) + alpha * cp.sum_squares(theta_t) \
            + cp.sum_squares(Ab @ theta_b - zb) + alpha * cp.sum_squares(theta_b)
        
        gap_shortfall = cp.pos(Delta - (A @ theta_t - A @ theta_b))
        obj_soft += 100.0 * cp.sum_squares(gap_shortfall)
        
        prob_soft = cp.Problem(cp.Minimize(obj_soft), [])
        
        try:
            prob_soft.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, max_iter=5000, verbose=False)
            if prob_soft.status in ["optimal", "optimal_inaccurate"]:
                print(f"Soft gap constraint succeeded with status: {prob_soft.status}")
                return theta_t.value, theta_b.value
        except Exception as e:
            last_error = f"Soft gap fallback also failed: {str(e)}"
    
    print(f"Warning: All QP solvers failed. Last error: {last_error}")
    return None, None
