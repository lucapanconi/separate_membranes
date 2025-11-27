#!/usr/bin/env python3
"""
Validation script for membrane separation algorithm.

Simulates synthetic membrane pairs and evaluates the performance of the membrane separator across a range of parameter combinations.
"""

# Imports
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product
import json
from typing import Dict, Tuple, List, Any
import warnings
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

from separate_membranes.alternating_fit import alternating_surface_fit
from separate_membranes.design_matrix import cubic_design_matrix
from separate_membranes.data_io import standardise_points

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Continue from last run instead of creating new folder
CONTINUE_LAST = False

D_MAX_VALUES = [50, 100, 150, 200, 250]
N_VALUES = [250, 500, 750, 1000]
DENSITY_AUGMENT_ORDERS = [1, 3, 5]
LAT_SIGMA_VALUES = [5, 10, 15, 20, 25]
AX_SIGMA_VALUES = [5, 10, 15, 20, 25]
DELTA_VALUES = [40, 80, 120, 160, 200]
OUTLIERS_VALUES = [5, 10, 15, 20, 25]
ITERATIONS = list(range(5))

# Default parameters
X_MAX = 3000
Y_MAX = 3000
GRID_SIZE = 100


# Set up parallel processing
TORCH_AVAILABLE = False
MULTIPROCESSING_AVAILABLE = False

try:
    import torch
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except ImportError:
    try:
        import multiprocessing as mp
        MULTIPROCESSING_AVAILABLE = True
    except:
        pass
    if not TORCH_AVAILABLE:
        warnings.warn("PyTorch not available. Will try standard multiprocessing.")


# Set up 3D plotting
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. 3D plots will be skipped.")


def density_augment(x: np.ndarray, order: int) -> np.ndarray:
    """Apply density augmentation function.
    
    Args:
        x: Input coordinates
        order: Order of polynomial
    
    Returns:
        Transformed coordinates
    """
    return x ** order


def generate_cubic_polynomial(x_max: float = X_MAX, y_max: float = Y_MAX) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random cubic polynomial for membrane surface.
    
    Returns:
        Tuple of (coefficients, polynomial function)
        Coefficients are in order: [1, x, y, x², xy, y², x³, x²y, xy², y³]
    """
    # Generate random coefficients from uniform [0, 1]
    coeffs = np.random.uniform(0, 1, 10)
    
    # Normalize coefficients by x_max and y_max proportional to order
    # Order: [1, x, y, x², xy, y², x³, x²y, xy², y³]
    normalization = np.array([
        1.0,  # 1
        1.0 / x_max,  # x
        1.0 / y_max,  # y
        1.0 / (x_max ** 2),  # x²
        1.0 / (x_max * y_max),  # xy
        1.0 / (y_max ** 2),  # y²
        1.0 / (x_max ** 3),  # x³
        1.0 / (x_max ** 2 * y_max),  # x²y
        1.0 / (x_max * y_max ** 2),  # xy²
        1.0 / (y_max ** 3)  # y³
    ])
    
    coeffs_normalized = coeffs * normalization
    
    def poly_func(x, y):
        """Evaluate polynomial at (x, y)."""
        A = cubic_design_matrix(x, y)
        return A @ coeffs_normalized
    
    return coeffs_normalized, poly_func


def evaluate_polynomial_on_grid(coeffs: np.ndarray, x_max: float, y_max: float, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Evaluate polynomial on a grid.
    
    Args:
        coeffs: Polynomial coefficients
        x_max: Maximum x value
        y_max: Maximum y value
        grid_size: Size of grid (grid_size x grid_size)
    
    Returns:
        Grid of z values
    """
    x_grid = np.linspace(0, x_max, grid_size)
    y_grid = np.linspace(0, y_max, grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    A = cubic_design_matrix(X_grid.ravel(), Y_grid.ravel())
    z_grid = (A @ coeffs).reshape(grid_size, grid_size)
    
    return z_grid


def run_single_simulation(params: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
    """Run a single simulation with given parameters.
    
    Args:
        params: Dictionary containing simulation parameters
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing results
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Read in parameters
    D_max = params['D_max']
    N = params['N']
    density_augment_order = params['density_augment_order']
    Lat_Sigma = params['Lat_Sigma']
    Ax_Sigma = params['Ax_Sigma']
    Delta = params['Delta']
    Outliers_pct = params['Outliers']
    iteration = params.get('iteration', 0)
    
    x_max = X_MAX
    y_max = Y_MAX
    
    # Generate upper membrane
    upper_coeffs, upper_poly = generate_cubic_polynomial(x_max, y_max)
    
    # Find min/max of polynomial on grid
    z_grid = evaluate_polynomial_on_grid(upper_coeffs, x_max, y_max)
    z_min = z_grid.min()
    z_max = z_grid.max()
    
    # Transform: min -> 0, max -> D_max
    scale = D_max / (z_max - z_min) if (z_max - z_min) > 0 else 1.0
    offset = -z_min * scale
    
    def transformed_upper_poly(x, y):
        """Transformed upper polynomial."""
        z = upper_poly(x, y)
        return z * scale + offset
    
    # Generate baseline points uniformly in [0, x_max] x [0, y_max]
    x_baseline = np.random.uniform(0, x_max, N)
    y_baseline = np.random.uniform(0, y_max, N)
    
    # Apply density augmentation to both x and y
    # Map all points to [0, 1] x [0, 1] range
    x_norm = x_baseline / x_max
    y_norm = y_baseline / y_max
    
    # Apply density augment function to normalized coordinates
    x_aug_norm = density_augment(x_norm, density_augment_order)
    y_aug_norm = density_augment(y_norm, density_augment_order)
    
    # Scale back to [0, x_max] x [0, y_max] range
    x_aug = x_aug_norm * x_max
    y_aug = y_aug_norm * y_max
    
    # Clamp
    x_aug = np.clip(x_aug, 0, x_max)
    y_aug = np.clip(y_aug, 0, y_max)
    
    # Get z coordinates from polynomial
    z_baseline = transformed_upper_poly(x_aug, y_aug)
    
    # Sample from 2D Gaussian for lateral spread
    upper_points = []
    for i in range(N):
        x_sample, y_sample = np.random.multivariate_normal(
            [x_aug[i], y_aug[i]], 
            [[Lat_Sigma**2, 0], [0, Lat_Sigma**2]]
        )
        z_sample = np.random.normal(z_baseline[i], Ax_Sigma)
        upper_points.append([x_sample, y_sample, z_sample])
    
    upper_points = np.array(upper_points)
    upper_labels = np.ones(N, dtype=int)  # 1 = upper
    
    # Create lower membrane by translating upper membrane down by Delta
    def transformed_lower_poly(x, y):
        """Lower membrane polynomial (translated down)."""
        return transformed_upper_poly(x, y) - Delta
    
    # Generate lower membrane points uniformly in [0, x_max] x [0, y_max]
    x_baseline_lower = np.random.uniform(0, x_max, N)
    y_baseline_lower = np.random.uniform(0, y_max, N)
    
    # Apply density augmentation to both x and y
    # Map all points to [0, 1] x [0, 1] range
    x_norm_lower = x_baseline_lower / x_max
    y_norm_lower = y_baseline_lower / y_max
    
    # Apply density augment function to normalized coordinates
    x_aug_norm_lower = density_augment(x_norm_lower, density_augment_order)
    y_aug_norm_lower = density_augment(y_norm_lower, density_augment_order)
    
    # Scale back to [0, x_max] x [0, y_max] range
    x_aug_lower = x_aug_norm_lower * x_max
    y_aug_lower = y_aug_norm_lower * y_max
    
    # Clamp
    x_aug_lower = np.clip(x_aug_lower, 0, x_max)
    y_aug_lower = np.clip(y_aug_lower, 0, y_max)
    
    z_baseline_lower = transformed_lower_poly(x_aug_lower, y_aug_lower)
    
    lower_points = []
    for i in range(N):
        x_sample, y_sample = np.random.multivariate_normal(
            [x_aug_lower[i], y_aug_lower[i]], 
            [[Lat_Sigma**2, 0], [0, Lat_Sigma**2]]
        )
        z_sample = np.random.normal(z_baseline_lower[i], Ax_Sigma)
        lower_points.append([x_sample, y_sample, z_sample])
    
    lower_points = np.array(lower_points)
    lower_labels = np.zeros(N, dtype=int)
    
    # Combine upper and lower points
    all_points = np.vstack([upper_points, lower_points])
    all_labels = np.hstack([upper_labels, lower_labels])
    
    # Generate outlier points
    n_outliers = int(np.ceil(2 * N * Outliers_pct / 100))
    x_min_all = all_points[:, 0].min()
    x_max_all = all_points[:, 0].max()
    y_min_all = all_points[:, 1].min()
    y_max_all = all_points[:, 1].max()
    z_min_all = all_points[:, 2].min()
    z_max_all = all_points[:, 2].max()
    
    outlier_points = np.column_stack([
        np.random.uniform(x_min_all, x_max_all, n_outliers),
        np.random.uniform(y_min_all, y_max_all, n_outliers),
        np.random.uniform(z_min_all, z_max_all, n_outliers)
    ])
    
    # Label outliers by closest membrane
    # Calculate distance to upper and lower surfaces at each outlier point
    outlier_labels = []
    for i in range(n_outliers):
        ox, oy, oz = outlier_points[i]
        z_upper = transformed_upper_poly(ox, oy)
        z_lower = transformed_lower_poly(ox, oy)
        dist_upper = abs(oz - z_upper)
        dist_lower = abs(oz - z_lower)
        outlier_labels.append(1 if dist_upper < dist_lower else 0)
    
    outlier_labels = np.array(outlier_labels)
    
    # Combine all points
    final_points = np.vstack([all_points, outlier_points])
    final_labels = np.hstack([all_labels, outlier_labels])
    
    # Store true polynomial coefficients (in original space, before transformation)
    true_upper_coeffs_transformed = upper_coeffs.copy()
    true_upper_coeffs_transformed[0] = upper_coeffs[0] * scale + offset
    true_upper_coeffs_transformed[1:] = upper_coeffs[1:] * scale
    
    true_lower_coeffs_transformed = true_upper_coeffs_transformed.copy()
    true_lower_coeffs_transformed[0] -= Delta
    
    # Run membrane separation
    x, y, z = final_points[:, 0], final_points[:, 1], final_points[:, 2]
    
    result = alternating_surface_fit(
        x, y, z,
        Delta=None,
        alpha=1e-3,
        max_iter=10,
        tol=0.01,
        soft_gap=True,
        lambda_gap=100.0,
        flat_surfaces=True
    )
    
    # Extract predictions
    pred_labels = result['labels_full']
    # Map: 1=upper, 0=lower, -1=outlier
    
    # Get found polynomial coefficients
    model = result['model']
    found_upper_coeffs = np.array(model['theta_top'])
    found_lower_coeffs = np.array(model['theta_bottom'])
    means = np.array(model['means'])
    stds = np.array(model['stds'])
    
    # Record how many outliers were detected
    n_outliers_detected = np.sum(pred_labels == -1)
    
    # Assign outliers to closest membrane surface
    outlier_indices = np.where(pred_labels == -1)[0]
    
    if len(outlier_indices) > 0:
        # Get outlier points
        outlier_points = final_points[outlier_indices]
        outlier_x = outlier_points[:, 0]
        outlier_y = outlier_points[:, 1]
        outlier_z = outlier_points[:, 2]
        
        # Normalize x, y coordinates for found polynomials
        outlier_x_norm = (outlier_x - means[0]) / stds[0]
        outlier_y_norm = (outlier_y - means[1]) / stds[1]
        
        # Evaluate found polynomial surfaces at outlier (x, y) coordinates
        A_outliers = cubic_design_matrix(outlier_x_norm, outlier_y_norm)
        z_upper_pred = (A_outliers @ found_upper_coeffs) * stds[2] + means[2]
        z_lower_pred = (A_outliers @ found_lower_coeffs) * stds[2] + means[2]
        
        # Calculate distance from each outlier point to each surface
        dist_to_upper = np.abs(outlier_z - z_upper_pred)
        dist_to_lower = np.abs(outlier_z - z_lower_pred)
        
        # Label outliers by closest membrane (1=upper, 0=lower)
        outlier_labels = (dist_to_upper <= dist_to_lower).astype(int)
        pred_labels[outlier_indices] = outlier_labels
    

    # Calculate Adjusted Rand Index
    # True: 1=upper, 0=lower
    # Pred: 1=upper, 0=lower
    ari = adjusted_rand_score(final_labels, pred_labels)
    
    # Calculate RMSE between true and found polynomials
    # Create grid for evaluation
    x_grid = np.linspace(0, x_max, GRID_SIZE)
    y_grid = np.linspace(0, y_max, GRID_SIZE)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Evaluate true polynomials
    A_true = cubic_design_matrix(X_grid.ravel(), Y_grid.ravel())
    z_true_upper = (A_true @ true_upper_coeffs_transformed).reshape(GRID_SIZE, GRID_SIZE)
    z_true_lower = (A_true @ true_lower_coeffs_transformed).reshape(GRID_SIZE, GRID_SIZE)
    
    # Denormalize found coefficients
    X_norm = (X_grid - means[0]) / stds[0]
    Y_norm = (Y_grid - means[1]) / stds[1]
    
    A_found = cubic_design_matrix(X_norm.ravel(), Y_norm.ravel())
    z_found_upper_norm = (A_found @ found_upper_coeffs).reshape(GRID_SIZE, GRID_SIZE)
    z_found_lower_norm = (A_found @ found_lower_coeffs).reshape(GRID_SIZE, GRID_SIZE)
    
    # Denormalize found z values
    z_found_upper = z_found_upper_norm * stds[2] + means[2]
    z_found_lower = z_found_lower_norm * stds[2] + means[2]
    
    # Calculate RMSE
    rmse_upper = np.sqrt(np.mean((z_true_upper - z_found_upper) ** 2))
    rmse_lower = np.sqrt(np.mean((z_true_lower - z_found_lower) ** 2))
    rmse_avg = (rmse_upper + rmse_lower) / 2
    
    # Calculate gap difference
    found_gap = model['Delta_norm'] * stds[2]
    gap_diff = abs(Delta - found_gap)
    
    # Store results
    results = {
        'D_max': D_max,
        'N': N,
        'density_augment_order': density_augment_order,
        'Lat_Sigma': Lat_Sigma,
        'Ax_Sigma': Ax_Sigma,
        'Delta': Delta,
        'Outliers': Outliers_pct,
        'iteration': iteration,
        'ARI': ari,
        'RMSE_upper': rmse_upper,
        'RMSE_lower': rmse_lower,
        'RMSE_avg': rmse_avg,
        'gap_diff': gap_diff,
        'true_gap': Delta,
        'found_gap': found_gap,
        'n_points': len(final_points),
        'n_upper_true': np.sum(final_labels == 1),
        'n_lower_true': np.sum(final_labels == 0),
        'n_upper_pred': np.sum(pred_labels == 1),
        'n_lower_pred': np.sum(pred_labels == 0),
        'n_outliers_detected': n_outliers_detected,
    }
    
    # Store point cloud and labels for visualization
    results['points'] = final_points
    results['true_labels'] = final_labels
    results['pred_labels'] = pred_labels
    results['true_upper_coeffs'] = true_upper_coeffs_transformed
    results['true_lower_coeffs'] = true_lower_coeffs_transformed
    results['found_upper_coeffs'] = found_upper_coeffs
    results['found_lower_coeffs'] = found_lower_coeffs
    results['means'] = means
    results['stds'] = stds
    results['x_max'] = x_max
    results['y_max'] = y_max
    
    return results


def run_with_seed(params):
    """Wrapper to add seed and handle errors for parallel processing."""
    try:
        # Generate specific seed from parameters
        seed = hash(str(params)) % (2**32)
        return run_single_simulation(params, seed=seed)
    except Exception as e:
        print(f"Error in simulation with params {params}: {e}")
        import traceback
        traceback.print_exc()
        result = params.copy()
        result.update({
            'ARI': np.nan,
            'RMSE_upper': np.nan,
            'RMSE_lower': np.nan,
            'RMSE_avg': np.nan,
            'gap_diff': np.nan,
            'error': str(e)
        })
        return result


def generate_parameter_combinations() -> pd.DataFrame:
    """Generate all parameter combinations for validation.
    
    Uses parameter ranges defined at top of script.
    
    Returns:
        DataFrame with all parameter combinations
    """
    combinations = list(product(
        D_MAX_VALUES, N_VALUES, DENSITY_AUGMENT_ORDERS,
        LAT_SIGMA_VALUES, AX_SIGMA_VALUES, DELTA_VALUES,
        OUTLIERS_VALUES, ITERATIONS
    ))
    
    df = pd.DataFrame(combinations, columns=[
        'D_max', 'N', 'density_augment_order', 'Lat_Sigma',
        'Ax_Sigma', 'Delta', 'Outliers', 'iteration'
    ])
    
    return df


def plot_3d_simulation(points: np.ndarray, true_labels: np.ndarray,
                       true_upper_coeffs: np.ndarray, true_lower_coeffs: np.ndarray,
                       x_max: float, y_max: float, title: str, filename: Path):
    """Plot 3D point cloud with true membrane surfaces."""
    if not PLOTLY_AVAILABLE:
        return
    
    # Color points by true label
    colors = ['red' if label == 1 else 'blue' for label in true_labels]
    
    fig = go.Figure()
    
    # Add points
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            opacity=0.6
        ),
        name='Points',
        text=[f'Label: {"upper" if l == 1 else "lower"}' for l in true_labels]
    ))
    
    # Add true surfaces
    grid_size = 30
    x_grid = np.linspace(0, x_max, grid_size)
    y_grid = np.linspace(0, y_max, grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    A = cubic_design_matrix(X_grid.ravel(), Y_grid.ravel())
    z_upper = (A @ true_upper_coeffs).reshape(grid_size, grid_size)
    z_lower = (A @ true_lower_coeffs).reshape(grid_size, grid_size)
    
    fig.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=z_upper,
        colorscale='Reds',
        opacity=0.5,
        showscale=False,
        name='Upper membrane'
    ))
    
    fig.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=z_lower,
        colorscale='Blues',
        opacity=0.5,
        showscale=False,
        name='Lower membrane'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000,
        height=800
    )
    
    fig.write_html(filename)
    return fig


def plot_3d_prediction(points: np.ndarray, pred_labels: np.ndarray,
                       found_upper_coeffs: np.ndarray, found_lower_coeffs: np.ndarray,
                       means: np.ndarray, stds: np.ndarray,
                       x_max: float, y_max: float, title: str, filename: Path):
    """Plot 3D point cloud with predicted membrane surfaces."""
    if not PLOTLY_AVAILABLE:
        return
    
    # Color points by prediction
    colors = []
    for label in pred_labels:
        if label == 1:
            colors.append('red')
        else:
            colors.append('blue')
    
    fig = go.Figure()
    
    # Add points
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            opacity=0.6
        ),
        name='Points'
    ))
    
    # Add predicted surfaces
    grid_size = 30
    x_grid = np.linspace(0, x_max, grid_size)
    y_grid = np.linspace(0, y_max, grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Normalize grid for coefficients
    X_norm = (X_grid - means[0]) / stds[0]
    Y_norm = (Y_grid - means[1]) / stds[1]
    
    A = cubic_design_matrix(X_norm.ravel(), Y_norm.ravel())
    z_upper_norm = (A @ found_upper_coeffs).reshape(grid_size, grid_size)
    z_lower_norm = (A @ found_lower_coeffs).reshape(grid_size, grid_size)
    
    # Denormalize
    z_upper = z_upper_norm * stds[2] + means[2]
    z_lower = z_lower_norm * stds[2] + means[2]
    
    fig.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=z_upper,
        colorscale='Reds',
        opacity=0.5,
        showscale=False,
        name='Upper membrane (predicted)'
    ))
    
    fig.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=z_lower,
        colorscale='Blues',
        opacity=0.5,
        showscale=False,
        name='Lower membrane (predicted)'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000,
        height=800
    )
    
    fig.write_html(filename)
    return fig


def create_summary_plots(results_df: pd.DataFrame, output_dir: Path):
    """Create summary plots for validation results."""
    # Set style
    plt.style.use('seaborn-v0_8-talk')
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Double the font sizes
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 22
    
    # Define colors for each parameter
    param_colors = {
        'D_max': '#1f77b4',          # Blue
        'N': '#ff7f0e',              # Orange
        'density_augment_order': '#2ca02c',  # Green
        'Lat_Sigma': '#d62728',      # Red
        'Ax_Sigma': '#9467bd',       # Purple
        'Delta': '#8c564b',          # Brown
        'Outliers': '#e377c2'        # Pink
    }
    
    # Define reference values for vertical lines
    reference_values = {
        'D_max': (184.65352159975933 + 150.75916131010277) / 2,
        'N': 945.7,
        'Delta': 100.72461722740061,
        'Outliers': 1 - (945.7 - 50.3) / 945.7
    }
    
    # Convert Outliers to percentage (0-100 scale) if needed
    reference_values['Outliers'] = reference_values['Outliers'] * 100
    
    # Parameters to plot
    param_names = ['D_max', 'N', 'density_augment_order', 'Lat_Sigma', 'Ax_Sigma', 'Delta', 'Outliers']
    
    for param in param_names:
        if param not in results_df.columns:
            continue
        
        # Get color for this parameter
        color = param_colors.get(param, '#1f77b4')
        
        # Group by parameter value
        grouped = results_df.groupby(param).agg({
            'ARI': ['mean', 'std'],
            'RMSE_avg': ['mean', 'std'],
            'gap_diff': ['mean', 'std']
        }).reset_index()
        
        grouped.columns = [param, 'ARI_mean', 'ARI_std', 'RMSE_mean', 'RMSE_std', 
                          'gap_diff_mean', 'gap_diff_std']
        
        # Plot ARI
        fig, ax = plt.subplots()
        ax.errorbar(grouped[param], grouped['ARI_mean'], yerr=grouped['ARI_std'],
                   marker='o', capsize=5, capthick=2, color=color, linewidth=2, markersize=8)
        ax.set_xlabel(param)
        ax.set_ylabel('Adjusted Rand Index')
        ax.set_title(f'ARI vs {param}')
        ax.set_ylim(0, 1)
        if param in reference_values:
            ax.axvline(x=reference_values[param], color='black', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'Reference: {reference_values[param]:.2f}')
            ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'ARI_vs_{param}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot RMSE
        fig, ax = plt.subplots()
        ax.errorbar(grouped[param], grouped['RMSE_mean'], yerr=grouped['RMSE_std'],
                   marker='o', capsize=5, capthick=2, color=color, linewidth=2, markersize=8)
        ax.set_xlabel(param)
        ax.set_ylabel('RMSE (averaged over both membranes)')
        ax.set_title(f'RMSE vs {param}')
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1])
        if param in reference_values:
            ax.axvline(x=reference_values[param], color='black', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'Reference: {reference_values[param]:.2f}')
            ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'RMSE_vs_{param}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot gap difference
        fig, ax = plt.subplots()
        ax.errorbar(grouped[param], grouped['gap_diff_mean'], yerr=grouped['gap_diff_std'],
                   marker='o', capsize=5, capthick=2, color=color, linewidth=2, markersize=8)
        ax.set_xlabel(param)
        ax.set_ylabel('Gap Difference (|true - found|)')
        ax.set_title(f'Gap Difference vs {param}')
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1])
        if param in reference_values:
            ax.axvline(x=reference_values[param], color='black', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'Reference: {reference_values[param]:.2f}')
            ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'GapDiff_vs_{param}.png', dpi=300, bbox_inches='tight')
        plt.close()


def save_single_result(result: Dict[str, Any], results_dir: Path):
    """Save a single simulation result to CSV."""
    # Create filename from parameters
    filename = (
        f"result_D{result['D_max']}_N{result['N']}_"
        f"DA{result['density_augment_order']}_"
        f"Lat{result['Lat_Sigma']}_Ax{result['Ax_Sigma']}_"
        f"Delta{result['Delta']}_Out{result['Outliers']}_"
        f"iter{result['iteration']}.csv"
    )
    
    # Save metrics
    exclude_keys = ['points', 'true_labels', 'pred_labels',
                   'true_upper_coeffs', 'true_lower_coeffs',
                   'found_upper_coeffs', 'found_lower_coeffs',
                   'means', 'stds', 'x_max', 'y_max']
    result_clean = {k: v for k, v in result.items() if k not in exclude_keys}
    result_df = pd.DataFrame([result_clean])
    result_df.to_csv(results_dir / filename, index=False)


def main():
    """Main function to run validation."""
    # Create validation folder
    validation_dir = Path('validation')
    validation_dir.mkdir(exist_ok=True)
    
    if CONTINUE_LAST:
        # Find most recent timestamped folder
        existing_folders = [f for f in validation_dir.iterdir() if f.is_dir() and len(f.name) == 15 and f.name[8] == '_']
        if existing_folders:
            # Sort by name to get most recent
            existing_folders.sort(key=lambda x: x.name, reverse=True)
            run_dir = existing_folders[0]
            print(f"Continuing in existing directory: {run_dir}")
        else:
            # No existing folder, create new one
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = validation_dir / timestamp
            run_dir.mkdir(exist_ok=True)
            print(f"No existing folder found. Creating new directory: {run_dir}")
    else:
        # Create new timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = validation_dir / timestamp
        run_dir.mkdir(exist_ok=True)
        print(f"Creating new validation run directory: {run_dir}")
    
    results_dir = run_dir / 'Results'
    results_dir.mkdir(exist_ok=True)
    
    plots_dir = run_dir / 'Plots'
    plots_dir.mkdir(exist_ok=True)
    
    print(f"Validation run directory: {run_dir}")
    print(f"Results will be saved to: {results_dir}")
    print(f"Plots will be saved to: {plots_dir}")
    
    # Generate parameter combinations
    print("\nGenerating parameter combinations...")
    param_df = generate_parameter_combinations()
    print(f"Total combinations: {len(param_df)}")
    
    # Save parameters
    params_file = run_dir / 'parameters.csv'
    if not params_file.exists():
        param_df.to_csv(params_file, index=False)
        print(f"Saved parameters to: {params_file}")
    else:
        print(f"Parameters file already exists: {params_file}")
    
    # Check existing results
    existing_files = list(results_dir.glob('result_*.csv'))
    existing_params = set()
    for f in existing_files:
        try:
            df_existing = pd.read_csv(f)
            if len(df_existing) > 0:
                # Extract parameter combination from first row
                row = df_existing.iloc[0]
                key = (
                    row['D_max'], row['N'], row['density_augment_order'],
                    row['Lat_Sigma'], row['Ax_Sigma'], row['Delta'],
                    row['Outliers'], row['iteration']
                )
                existing_params.add(key)
        except (KeyError, pd.errors.EmptyDataError, ValueError) as e:
            # Skip files that can't be read or don't have required columns
            continue
    
    # Filter out existing combinations
    param_df['key'] = list(zip(
        param_df['D_max'], param_df['N'], param_df['density_augment_order'],
        param_df['Lat_Sigma'], param_df['Ax_Sigma'], param_df['Delta'],
        param_df['Outliers'], param_df['iteration']
    ))
    
    param_df = param_df[~param_df['key'].isin(existing_params)]
    param_df = param_df.drop(columns=['key'])
    
    print(f"Remaining combinations to simulate: {len(param_df)}")
    
    if len(param_df) == 0:
        print("All combinations already simulated. Exiting.")
        return
    
    # Convert to list of dictionaries for parallel processing
    param_list = param_df.to_dict('records')
    
    # Run simulations
    print("\nRunning simulations...")
    
    results = []
    
    if (TORCH_AVAILABLE or MULTIPROCESSING_AVAILABLE) and len(param_list) > 1:
        # Use parallel processing
        if TORCH_AVAILABLE:
            print("Using parallel processing with torch.multiprocessing...")
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
            
            from torch.multiprocessing import Pool
            with Pool() as pool:
                # Use imap_unordered to process results as they complete
                for i, result in enumerate(pool.imap_unordered(run_with_seed, param_list), 1):
                    results.append(result)
                    # Save
                    save_single_result(result, results_dir)
                    if i % 10 == 0 or i == len(param_list):
                        print(f"  Progress: {i}/{len(param_list)} simulations completed and saved")
        else:
            print("Using parallel processing with multiprocessing...")
            from multiprocessing import Pool
            with Pool() as pool:
                # Use imap_unordered to process results as they complete
                for i, result in enumerate(pool.imap_unordered(run_with_seed, param_list), 1):
                    results.append(result)
                    # Save
                    save_single_result(result, results_dir)
                    if i % 10 == 0 or i == len(param_list):
                        print(f"  Progress: {i}/{len(param_list)} simulations completed and saved")
    else:
        # Sequential processing with immediate saving
        print("Using sequential processing...")
        for i, params in enumerate(param_list, 1):
            result = run_with_seed(params)
            results.append(result)
            # Save
            save_single_result(result, results_dir)
            if i % 10 == 0 or i == len(param_list):
                print(f"  Progress: {i}/{len(param_list)} simulations completed and saved")
    
    print(f"\nCompleted {len(results)} simulations")
    
    # Combine all results
    print("\nCombining results...")
    all_results_clean = []
    exclude_keys = ['points', 'true_labels', 'pred_labels',
                   'true_upper_coeffs', 'true_lower_coeffs',
                   'found_upper_coeffs', 'found_lower_coeffs',
                   'means', 'stds', 'x_max', 'y_max']
    for result in results:
        result_clean = {k: v for k, v in result.items() 
                       if k not in exclude_keys}
        all_results_clean.append(result_clean)
    
    combined_df = pd.DataFrame(all_results_clean)
    combined_df.to_csv(run_dir / 'combined_results.csv', index=False)
    print(f"Saved combined results to: {run_dir / 'combined_results.csv'}")
    
    # Create summary plots
    print("\nCreating summary plots...")
    create_summary_plots(combined_df, plots_dir)
    print(f"Summary plots saved to: {plots_dir}")
    
    # Select 8 random simulations for detailed visualization
    print("\nCreating detailed visualizations for 8 random simulations...")
    # Filter out failed simulations
    valid_results = [r for r in results if 'points' in r and 'error' not in r]
    
    if len(valid_results) > 0:
        selected_indices = np.random.choice(len(valid_results), min(8, len(valid_results)), replace=False)
        
        for idx in selected_indices:
            result = valid_results[idx]
            sim_id = (
                f"D{result['D_max']}_N{result['N']}_"
                f"DA{result['density_augment_order']}_"
                f"Lat{result['Lat_Sigma']}_Ax{result['Ax_Sigma']}_"
                f"Delta{result['Delta']}_Out{result['Outliers']}_"
                f"iter{result['iteration']}"
            )
            
            try:
                # Plot true labels
                plot_3d_simulation(
                    result['points'],
                    result['true_labels'],
                    result['true_upper_coeffs'],
                    result['true_lower_coeffs'],
                    result['x_max'],
                    result['y_max'],
                    f'True Labels - {sim_id}',
                    plots_dir / f'true_labels_{sim_id}.html'
                )
                
                # Plot predictions
                plot_3d_prediction(
                    result['points'],
                    result['pred_labels'],
                    result['found_upper_coeffs'],
                    result['found_lower_coeffs'],
                    result['means'],
                    result['stds'],
                    result['x_max'],
                    result['y_max'],
                    f'Predictions - {sim_id}',
                    plots_dir / f'predictions_{sim_id}.html'
                )
            except Exception as e:
                print(f"Warning: Could not create visualization for {sim_id}: {e}")
    else:
        print("No valid results available for visualization.")
    
    print(f"Detailed visualizations saved to: {plots_dir}")
    print(f"\nValidation complete! Results in: {run_dir}")


if __name__ == '__main__':
    main()

