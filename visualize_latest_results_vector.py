#!/usr/bin/env python3
"""
Script to create vector graphics visualisations of membrane separation results.

This script creates vector graphics (SVG/PDF) of the point clouds and membrane surfaces.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors

from separate_membranes.design_matrix import cubic_design_matrix


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

PLOT_CONFIG = {
    'figure_width': 12.3,
    'figure_height': 9,
    'point_size': 4,
    'point_alpha': 0.7,
    'surface_alpha': 0.6,
    'grid_resolution': 50,
    'colors': {
        'upper': '#ef4444',
        'lower': '#3b82f6',
    },
    'view_azim': 45,
    'view_elev': 15,
    'dpi': 300,
}


def find_latest_results_folder(results_dir: Path = Path("results")) -> Optional[Path]:
    """Find the most recent timestamped results folder.
    
    Args:
        results_dir: Base results directory
        
    Returns:
        Path to latest timestamped folder, or None if not found
    """
    if not results_dir.exists():
        return None
    
    # Find folders with timestamp format YYYYMMDD_HHMMSS
    timestamped_folders = [
        f for f in results_dir.iterdir() 
        if f.is_dir() and len(f.name) == 15 and f.name[8] == '_'
    ]
    
    if not timestamped_folders:
        return None
    
    # Sort by name (timestamp) and return most recent
    timestamped_folders.sort(key=lambda x: x.name, reverse=True)
    return timestamped_folders[0]


def load_result_data(result_folder: Path, file_stem: str) -> Dict[str, Any]:
    """Load all data for a specific result file.
    
    Args:
        result_folder: Path to timestamped results folder
        file_stem: Stem of the file (without extension)
        
    Returns:
        Dictionary containing points, labels, and model parameters
    """
    # Load labeled CSV
    csv_path = result_folder / f"{file_stem}_labeled.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Labeled CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    points = df[['mx', 'my', 'mz']].values
    
    # Map string labels to numeric
    label_map = {'upper': 1, 'lower': 0, 'outlier': -1}
    labels = np.array([label_map[label] for label in df['membrane']])
    
    # Load model parameters
    json_path = result_folder / f"{file_stem}_metrics.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {json_path}")
    
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    model = metrics['model']
    theta_top = np.array(model['theta_top'])
    theta_bottom = np.array(model['theta_bottom'])
    means = np.array(model['means'])
    stds = np.array(model['stds'])
    
    return {
        'points': points,
        'labels': labels,
        'theta_top': theta_top,
        'theta_bottom': theta_bottom,
        'means': means,
        'stds': stds,
        'file_name': file_stem
    }


def create_surface_mesh(theta: np.ndarray, means: np.ndarray, stds: np.ndarray,
                       x_min: float, x_max: float, y_min: float, y_max: float,
                       grid_res: int, x_shift: float = 0, y_shift: float = 0) -> tuple:
    """Create mesh for membrane surface.
    
    Args:
        theta: Polynomial coefficients (in normalized space)
        means: Normalization means [x, y, z]
        stds: Normalization standard deviations [x, y, z]
        x_min, x_max, y_min, y_max: Bounds of the surface (after shifting)
        grid_res: Number of grid points in each dimension
        x_shift, y_shift: Amount to shift coordinates by
        
    Returns:
        Tuple of (X_grid, Y_grid, Z_grid) for surface
    """
    # Create grid in shifted space
    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Convert back to original space for normalisation
    X_grid_orig = X_grid + x_shift
    Y_grid_orig = Y_grid + y_shift
    
    # Normalise grid coordinates
    X_norm = (X_grid_orig - means[0]) / stds[0]
    Y_norm = (Y_grid_orig - means[1]) / stds[1]
    
    # Evaluate polynomial at grid points
    A = cubic_design_matrix(X_norm.ravel(), Y_norm.ravel())
    z_norm = A @ theta
    
    # Denormalise z values
    Z_grid = (z_norm.reshape(grid_res, grid_res) * stds[2] + means[2])
    
    return X_grid, Y_grid, Z_grid


def plot_membranes_3d_vector(data: Dict[str, Any], config: Dict[str, Any] = None,
                             output_path: Optional[Path] = None):
    """Create vector graphics plot of point cloud and membrane surfaces.
    
    Args:
        data: Dictionary containing points, labels, and model parameters
        config: Plot configuration dictionary (uses PLOT_CONFIG if None)
        output_path: Path to save image file (without extension)
    """
    if config is None:
        config = PLOT_CONFIG
    
    points = data['points'].copy()
    labels = data['labels']
    theta_top = data['theta_top']
    theta_bottom = data['theta_bottom']
    means = data['means']
    stds = data['stds']
    file_name = data['file_name']
    
    # Shift points so minimum x and y are at 0
    x_shift = points[:, 0].min()
    y_shift = points[:, 1].min()
    points[:, 0] -= x_shift
    points[:, 1] -= y_shift
    
    # Get bounds for surfaces (after shifting)
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # Assign outliers to closest membrane
    labels_colored = labels.copy()
    outlier_mask = labels == -1
    
    if np.any(outlier_mask):
        outlier_points = points[outlier_mask]
        outlier_x = outlier_points[:, 0] + x_shift
        outlier_y = outlier_points[:, 1] + y_shift
        outlier_z = outlier_points[:, 2]
        
        # Normalise coordinates
        outlier_x_norm = (outlier_x - means[0]) / stds[0]
        outlier_y_norm = (outlier_y - means[1]) / stds[1]
        
        # Evaluate surfaces at outlier positions
        A_outliers = cubic_design_matrix(outlier_x_norm, outlier_y_norm)
        z_upper_pred = (A_outliers @ theta_top) * stds[2] + means[2]
        z_lower_pred = (A_outliers @ theta_bottom) * stds[2] + means[2]
        
        # Calculate distances
        dist_to_upper = np.abs(outlier_z - z_upper_pred)
        dist_to_lower = np.abs(outlier_z - z_lower_pred)
        
        # Assign outliers to closest membrane
        outlier_labels = (dist_to_upper <= dist_to_lower).astype(int)
        labels_colored[outlier_mask] = outlier_labels
    
    upper_mask = labels_colored == 1
    lower_mask = labels_colored == 0
    
    # Create figure
    fig = plt.figure(figsize=(config['figure_width'], config['figure_height']))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(points[upper_mask, 0], points[upper_mask, 1], points[upper_mask, 2],
               c=config['colors']['upper'], s=config['point_size'], 
               alpha=config['point_alpha'], depthshade=True)
    
    ax.scatter(points[lower_mask, 0], points[lower_mask, 1], points[lower_mask, 2],
               c=config['colors']['lower'], s=config['point_size'],
               alpha=config['point_alpha'], depthshade=True)
    
    # Create and plot surfaces
    X_top, Y_top, Z_top = create_surface_mesh(
        theta_top, means, stds, x_min, x_max, y_min, y_max,
        config['grid_resolution'], x_shift, y_shift
    )
    
    X_bottom, Y_bottom, Z_bottom = create_surface_mesh(
        theta_bottom, means, stds, x_min, x_max, y_min, y_max,
        config['grid_resolution'], x_shift, y_shift
    )
    
    ax.plot_surface(X_top, Y_top, Z_top, color=config['colors']['upper'],
                   alpha=config['surface_alpha'], shade=True,
                   linewidth=0, antialiased=True)
    
    ax.plot_surface(X_bottom, Y_bottom, Z_bottom, color=config['colors']['lower'],
                   alpha=config['surface_alpha'], shade=True,
                   linewidth=0, antialiased=True)
    
    # Set labels with font for axis titles (x and y first)
    ax.set_xlabel('x', fontsize=20, labelpad=15, fontweight='bold')
    ax.set_ylabel('y', fontsize=20, labelpad=15, fontweight='bold')
    
    # Set tick label size
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.tick_params(axis='z', which='major', labelsize=17, pad=10)
    
    # Set axis limits to show full data range
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Set custom ticks for x and y axes
    x_ticks = [t for t in [0, 1000, 2000, 3000] if x_min <= t <= x_max]
    y_ticks = [t for t in [0, 1000, 2000, 3000] if y_min <= t <= y_max]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Set aspect ratio
    ax.set_box_aspect([1.6, 1.6, 0.625])
    
    # Set view angle
    ax.view_init(elev=config['view_elev'], azim=config['view_azim'])
    
    # Style the axes
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    
    # Remove black axis lines
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    # Set grid color to light grey
    ax.xaxis._axinfo["grid"]['color'] = '#999999'
    ax.yaxis._axinfo["grid"]['color'] = '#999999'
    ax.zaxis._axinfo["grid"]['color'] = '#999999'
    
    # Set background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    
    # Add manual z-axis label
    z_min, z_max = ax.get_zlim()
    z_mid = (z_min + z_max) / 2
    ax.text(x_max + (x_max - x_min) * 0.1, y_min - (y_max - y_min) * 0.1, z_mid, 
            'z', fontsize=20, fontweight='bold', 
            ha='center', va='center')
    
    # Save as vector graphics
    if output_path:
        # Save as SVG with tight bounding box and padding
        svg_path = output_path.with_suffix('.svg')
        plt.savefig(svg_path, format='svg', dpi=config['dpi'],
                   bbox_inches='tight', pad_inches=0.3,
                   facecolor='white', edgecolor='none')
        print(f"  SVG saved: {svg_path}")
        
        # Save as PDF with tight bounding box and padding
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, format='pdf', dpi=config['dpi'],
                   bbox_inches='tight', pad_inches=0.3,
                   facecolor='white', edgecolor='none')
        print(f"  PDF saved: {pdf_path}")
        
        # Also save as high-resolution PNG
        png_path = output_path.with_suffix('.png')
        plt.savefig(png_path, format='png', dpi=config['dpi'],
                   bbox_inches='tight', pad_inches=0.3,
                   facecolor='white', edgecolor='none')
        print(f"  PNG saved: {png_path}")
    
    plt.close(fig)


def main():
    """Main function to create vector graphics visualizations."""
    # Find latest results folder
    results_dir = Path('results')
    latest_folder = find_latest_results_folder(results_dir)
    
    if latest_folder is None:
        print(f"No timestamped results folders found in {results_dir}")
        return
    
    print(f"Found latest results folder: {latest_folder}")
    
    # Set output directory using the same timestamp
    timestamp = latest_folder.name
    single_plots_dir = Path('single_plots')
    single_plots_dir.mkdir(exist_ok=True)
    output_dir = single_plots_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Use config from PLOT_CONFIG
    config = PLOT_CONFIG.copy()
    
    # Find all labeled CSV files
    csv_files = list(latest_folder.glob('*_labeled.csv'))
    
    if not csv_files:
        print(f"No labeled CSV files found in {latest_folder}")
        return
    
    print(f"Found {len(csv_files)} file(s) to visualize")
    
    # Process each file
    for csv_file in csv_files:
        file_stem = csv_file.stem.replace('_labeled', '')
        print(f"\nProcessing: {file_stem}")
        
        try:
            # Load data
            data = load_result_data(latest_folder, file_stem)
            
            # Create static vector graphics plot
            output_path = output_dir / file_stem
            plot_membranes_3d_vector(data, config, output_path)
            
            print(f"  ✓ Vector graphics created")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nVector graphics generation complete! Files saved to: {output_dir}")


if __name__ == '__main__':
    main()

