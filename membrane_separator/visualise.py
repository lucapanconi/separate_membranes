import plotly.graph_objects as go
import numpy as np
from .design_matrix import cubic_design_matrix

def plot_3d_points_by_z(x, y, z, title, filename, plot_dir):
    """Plot 3D points colored by z-height."""
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=z,
            colorscale='viridis',
            opacity=0.8,
            colorbar=dict(title="Z Height")
        ),
        name='Points'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=600
    )
    
    fig.write_html(plot_dir / f"{filename}.html")
    return fig


def plot_3d_with_surfaces(x, y, z, labels, θt, θb, means, stds, title, filename, plot_dir):
    """Plot 3D points with fitted surfaces."""
    # Denormalize points for display
    pts_orig = np.column_stack([x, y, z]) * stds + means
    x_orig, y_orig, z_orig = pts_orig[:, 0], pts_orig[:, 1], pts_orig[:, 2]
    
    # Create surface grids
    x_min, x_max = x_orig.min(), x_orig.max()
    y_min, y_max = y_orig.min(), y_orig.max()
    
    x_grid = np.linspace(x_min, x_max, 30)
    y_grid = np.linspace(y_min, y_max, 30)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Normalize grid points
    X_norm = (X_grid - means[0]) / stds[0]
    Y_norm = (Y_grid - means[1]) / stds[1]
    
    # Evaluate surfaces
    A_grid = cubic_design_matrix(X_norm.ravel(), Y_norm.ravel())
    Z_top = (A_grid @ θt).reshape(X_grid.shape)
    Z_bottom = (A_grid @ θb).reshape(X_grid.shape)
    
    # Denormalize Z coordinates
    Z_top = Z_top * stds[2] + means[2]
    Z_bottom = Z_bottom * stds[2] + means[2]
    
    fig = go.Figure()
    
    # Add points colored by surface assignment
    colors = ['red' if l == 1 else 'blue' for l in labels]
    fig.add_trace(go.Scatter3d(
        x=x_orig, y=y_orig, z=z_orig,
        mode='markers',
        marker=dict(size=2, color=colors, opacity=0.6),
        name='Points'
    ))
    
    # Add top surface
    fig.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=Z_top,
        colorscale='Reds',
        opacity=0.3,
        name='Top Surface'
    ))
    
    # Add bottom surface
    fig.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=Z_bottom,
        colorscale='Blues',
        opacity=0.3,
        name='Bottom Surface'
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
    
    fig.write_html(plot_dir / f"{filename}.html")
    return fig


def plot_3d_by_distance_to_manifolds(x, y, z, labels, θt, θb, means, stds, title, filename, plot_dir):
    """Plot 3D points colored by distance to nearest manifold."""
    # Denormalize points for display
    pts_orig = np.column_stack([x, y, z]) * stds + means
    x_orig, y_orig, z_orig = pts_orig[:, 0], pts_orig[:, 1], pts_orig[:, 2]
    
    # Compute distances to both surfaces
    A = cubic_design_matrix(x, y)
    z_top_pred = A @ θt
    z_bottom_pred = A @ θb
    
    # Denormalize predictions
    z_top_pred = z_top_pred * stds[2] + means[2]
    z_bottom_pred = z_bottom_pred * stds[2] + means[2]
    
    # Compute distances
    dist_to_top = np.abs(z_orig - z_top_pred)
    dist_to_bottom = np.abs(z_orig - z_bottom_pred)
    min_distances = np.minimum(dist_to_top, dist_to_bottom)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_orig, y=y_orig, z=z_orig,
        mode='markers',
        marker=dict(
            size=2,
            color=min_distances,
            colorscale='plasma',
            opacity=0.8,
            colorbar=dict(title="Distance to Nearest Surface")
        ),
        name='Points'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=600
    )
    
    fig.write_html(plot_dir / f"{filename}.html")
    return fig


def plot_3d_by_classification(x, y, z, labels, title, filename, plot_dir):
    """Plot 3D points colored by classification (upper/lower/outlier)."""
    # Denormalize points for display (assuming they're already normalized)
    fig = go.Figure()
    
    # Separate points by classification
    upper_mask = labels == 1
    lower_mask = labels == 0
    outlier_mask = labels == -1
    
    if np.any(upper_mask):
        fig.add_trace(go.Scatter3d(
            x=x[upper_mask], y=y[upper_mask], z=z[upper_mask],
            mode='markers',
            marker=dict(size=3, color='red', opacity=0.8),
            name='Upper Membrane'
        ))
    
    if np.any(lower_mask):
        fig.add_trace(go.Scatter3d(
            x=x[lower_mask], y=y[lower_mask], z=z[lower_mask],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.8),
            name='Lower Membrane'
        ))
    
    if np.any(outlier_mask):
        fig.add_trace(go.Scatter3d(
            x=x[outlier_mask], y=y[outlier_mask], z=z[outlier_mask],
            mode='markers',
            marker=dict(size=2, color='gray', opacity=0.5),
            name='Outliers'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=600
    )
    
    fig.write_html(plot_dir / f"{filename}.html")
    return fig


def plot_combined_dashboard(x, y, z, labels, θt, θb, means, stds, title, filename, plot_dir):
    """Create a combined interactive dashboard with all four visualizations."""
    from plotly.subplots import make_subplots
    
    # Denormalize points for display
    pts_orig = np.column_stack([x, y, z]) * stds + means
    x_orig, y_orig, z_orig = pts_orig[:, 0], pts_orig[:, 1], pts_orig[:, 2]
    
    # Create surface grids for the surfaces plot
    x_min, x_max = x_orig.min(), x_orig.max()
    y_min, y_max = y_orig.min(), y_orig.max()
    
    x_grid = np.linspace(x_min, x_max, 20)  # Reduced resolution for performance
    y_grid = np.linspace(y_min, y_max, 20)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Normalize grid points
    X_norm = (X_grid - means[0]) / stds[0]
    Y_norm = (Y_grid - means[1]) / stds[1]
    
    # Evaluate surfaces
    A_grid = cubic_design_matrix(X_norm.ravel(), Y_norm.ravel())
    Z_top = (A_grid @ θt).reshape(X_grid.shape)
    Z_bottom = (A_grid @ θb).reshape(X_grid.shape)
    
    # Denormalize Z coordinates
    Z_top = Z_top * stds[2] + means[2]
    Z_bottom = Z_bottom * stds[2] + means[2]
    
    # Compute distances for the distance plot
    A = cubic_design_matrix(x, y)
    z_top_pred = A @ θt
    z_bottom_pred = A @ θb
    
    # Denormalize predictions
    z_top_pred = z_top_pred * stds[2] + means[2]
    z_bottom_pred = z_bottom_pred * stds[2] + means[2]
    
    # Compute distances
    dist_to_top = np.abs(z_orig - z_top_pred)
    dist_to_bottom = np.abs(z_orig - z_bottom_pred)
    min_distances = np.minimum(dist_to_top, dist_to_bottom)
    
    # Create subplots with tighter spacing
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Original by Z-Height', 'With Fitted Surfaces', 
                       'Distance to Nearest Surface', 'Surface Classification'),
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    # Plot 1: Original by Z-height
    fig.add_trace(
        go.Scatter3d(
            x=x_orig, y=y_orig, z=z_orig,
            mode='markers',
            marker=dict(
                size=1.5,
                color=z_orig,
                colorscale='viridis',
                opacity=0.8,
                colorbar=dict(x=1.02, y=0.8, len=0.3, thickness=15, title="Z Height")
            ),
            name='Original Points',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Plot 2: With fitted surfaces
    # Add points
    colors = ['red' if l == 1 else 'blue' for l in labels]
    fig.add_trace(
        go.Scatter3d(
            x=x_orig, y=y_orig, z=z_orig,
            mode='markers',
            marker=dict(size=1.5, color=colors, opacity=0.6),
            name='Points',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add top surface
    fig.add_trace(
        go.Surface(
            x=X_grid, y=Y_grid, z=Z_top,
            colorscale='Reds',
            opacity=0.3,
            name='Top Surface',
            showlegend=False,
            showscale=False  # Disable colorbar for this surface
        ),
        row=1, col=2
    )
    
    # Add bottom surface
    fig.add_trace(
        go.Surface(
            x=X_grid, y=Y_grid, z=Z_bottom,
            colorscale='Blues',
            opacity=0.3,
            name='Bottom Surface',
            showlegend=False,
            showscale=False  # Disable colorbar for this surface
        ),
        row=1, col=2
    )
    
    # Plot 3: Distance to nearest surface
    fig.add_trace(
        go.Scatter3d(
            x=x_orig, y=y_orig, z=z_orig,
            mode='markers',
            marker=dict(
                size=1.5,
                color=min_distances,
                colorscale='plasma',
                opacity=0.8,
                colorbar=dict(x=1.02, y=0.2, len=0.3, thickness=15, title="Distance")
            ),
            name='Distance',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Plot 4: Classification
    # Separate points by classification
    upper_mask = labels == 1
    lower_mask = labels == 0
    outlier_mask = labels == -1
    
    if np.any(upper_mask):
        fig.add_trace(
            go.Scatter3d(
                x=x_orig[upper_mask], y=y_orig[upper_mask], z=z_orig[upper_mask],
                mode='markers',
                marker=dict(size=2, color='red', opacity=0.8),
                name='Upper Membrane',
                showlegend=False
            ),
            row=2, col=2
        )
    
    if np.any(lower_mask):
        fig.add_trace(
            go.Scatter3d(
                x=x_orig[lower_mask], y=y_orig[lower_mask], z=z_orig[lower_mask],
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.8),
                name='Lower Membrane',
                showlegend=False
            ),
            row=2, col=2
        )
    
    if np.any(outlier_mask):
        fig.add_trace(
            go.Scatter3d(
                x=x_orig[outlier_mask], y=y_orig[outlier_mask], z=z_orig[outlier_mask],
                mode='markers',
                marker=dict(size=1.5, color='gray', opacity=0.5),
                name='Outliers',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout - more compact for Chrome with space for colorbars
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=14)
        ),
        height=800,
        width=1400,  # Increased width to accommodate colorbars
        showlegend=False,
        margin=dict(l=20, r=80, t=60, b=20)  # More right margin for colorbars
    )
    
    # Update scene properties for all subplots
    scenes = ['scene', 'scene2', 'scene3', 'scene4']
    for scene in scenes:
        fig.update_layout(**{
            scene: dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        })
    
    # Save the combined plot
    fig.write_html(plot_dir / f"{filename}.html")
    return fig