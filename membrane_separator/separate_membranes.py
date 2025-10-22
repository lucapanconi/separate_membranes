import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
from typing import Union, Dict, Any

from .data_io import load_point_cloud
from .alternating_fit import alternating_surface_fit


def separate_membranes(
    data: Union[str, Path, pd.DataFrame],
    results_dir: Union[str, Path] = "results",
    plots_dir: Union[str, Path] = "plots",
    generate_plots: bool = True,
    combined_dashboard: bool = True,
    flat_surfaces: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Separate membrane surfaces from 3D point cloud data.
    
    Args:
        data (Union[str, Path, pd.DataFrame]): The data to process. Can be a string path to a directory containing CSV files, a Path object, or a pandas DataFrame.
        results_dir (Union[str, Path]): The directory to save the results. Defaults to "results".
        plots_dir (Union[str, Path]): The directory to save the plots. Defaults to "plots".
        generate_plots (bool): Whether to generate plots. Defaults to True.
        combined_dashboard (bool): Whether to generate a combined dashboard of all plots. Defaults to True.
        flat_surfaces (bool): Whether to try enforcing flat surfaces. This places stronger regularisation on high-order coefficients in the surface polynomial and tries to force the boundary of the surface to approximate the mean z-value of the data. Defaults to True.
        **kwargs (Any): Additional keyword arguments.
    """
    
    default_params = {
        'max_iter': 10,
        'tol': 0.01,
        'alpha': 1e-3,
        'soft_gap': True,
        'lambda_gap': 100.0,
        'delta': None
    }
    params = {**default_params, **kwargs}
    
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_timestamped = results_dir / timestamp
    plots_timestamped = plots_dir / timestamp if generate_plots else None
    
    results_timestamped.mkdir(parents=True, exist_ok=True)
    if plots_timestamped:
        plots_timestamped.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {results_timestamped}")
    if plots_timestamped:
        print(f"Plots will be saved to: {plots_timestamped}")
    
    processed_files = []
    summary_stats = {
        'total_points': 0,
        'total_upper': 0,
        'total_lower': 0,
        'total_outliers': 0,
        'files_processed': 0,
        'errors': []
    }
    
    if isinstance(data, pd.DataFrame):
        csv_files = [('dataframe', data)]
    else:
        data_path = Path(data)
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        csv_files = list(data_path.rglob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in: {data_path}")
        
        print(f"Found {len(csv_files)} CSV file(s) in {data_path}")
        csv_files = [(f.name, f) for f in csv_files]
    
    for file_name, file_path in csv_files:
        try:
            print(f"\nProcessing: {file_name}")
            
            if isinstance(file_path, pd.DataFrame):
                df = file_path.copy()
                pts = df[["mx", "my", "mz"]].to_numpy(dtype=float)
            else:
                pts = load_point_cloud(str(file_path))
                df = pd.read_csv(file_path)
            
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            
            result = alternating_surface_fit(
                x, y, z,
                Delta=params['delta'],
                alpha=params['alpha'],
                max_iter=params['max_iter'],
                tol=params['tol'],
                soft_gap=params['soft_gap'],
                lambda_gap=params['lambda_gap'],
                flat_surfaces=flat_surfaces
            )
            
            labels_full = result["labels_full"]
            label_map = {1: "upper", 0: "lower", -1: "outlier"}
            df["membrane"] = [label_map[int(v)] for v in labels_full]
            
            base_name = Path(file_name).stem if file_name != 'dataframe' else 'dataframe'
            
            out_csv = results_timestamped / f"{base_name}_labeled.csv"
            df.to_csv(out_csv, index=False)
            
            model = result["model"]
            metrics = result["metrics"]
            summary = {
                "file": file_name,
                "model": model,
                "metrics": metrics,
                "parameters": params
            }
            out_json = results_timestamped / f"{base_name}_metrics.json"
            with open(out_json, "w") as fp:
                json.dump(summary, fp, indent=2)
            
            np.save(results_timestamped / f"{base_name}_theta_top.npy", np.array(model["theta_top"]))
            np.save(results_timestamped / f"{base_name}_theta_bottom.npy", np.array(model["theta_bottom"]))
            
            if generate_plots and plots_timestamped:
                _generate_plots(
                    x, y, z, result, model, base_name, plots_timestamped,
                    combined_dashboard
                )
            
            processed_files.append(file_name)
            summary_stats['files_processed'] += 1
            summary_stats['total_points'] += len(df)
            summary_stats['total_upper'] += np.sum(labels_full == 1)
            summary_stats['total_lower'] += np.sum(labels_full == 0)
            summary_stats['total_outliers'] += np.sum(labels_full == -1)
            
            print(f"  ✓ Saved: {out_csv.name}, {out_json.name}")
            print(f"  Points: {len(df)} total, {np.sum(labels_full >= 0)} kept, {np.sum(labels_full == -1)} outliers")
            print(f"  Upper: {np.sum(labels_full == 1)}, Lower: {np.sum(labels_full == 0)}")
            print(f"  RMSE (norm): top={metrics['RMSE_top_norm']:.4f}, bottom={metrics['RMSE_bottom_norm']:.4f}")
            
        except Exception as e:
            error_msg = f"ERROR processing {file_name}: {e}"
            print(error_msg)
            summary_stats['errors'].append(error_msg)
            warnings.warn(error_msg)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {summary_stats['files_processed']}")
    print(f"Total points: {summary_stats['total_points']}")
    print(f"Upper membrane: {summary_stats['total_upper']}")
    print(f"Lower membrane: {summary_stats['total_lower']}")
    print(f"Outliers: {summary_stats['total_outliers']}")
    if summary_stats['errors']:
        print(f"Errors: {len(summary_stats['errors'])}")
    print(f"Results saved to: {results_timestamped}")
    if plots_timestamped:
        print(f"Plots saved to: {plots_timestamped}")
    
    return {
        'results_dir': results_timestamped,
        'plots_dir': plots_timestamped,
        'processed_files': processed_files,
        'summary': summary_stats
    }


def _generate_plots(x, y, z, result, model, base_name, plots_dir, combined_dashboard):
    try:
        from .visualise import (
            plot_3d_points_by_z,
            plot_3d_with_surfaces,
            plot_3d_by_distance_to_manifolds,
            plot_3d_by_classification,
            plot_combined_dashboard
        )
        
        filtered_pts = result["filtered_points"]
        labels_filtered = result["labels_filtered"]
        x_filt, y_filt, z_filt = filtered_pts[:, 0], filtered_pts[:, 1], filtered_pts[:, 2]
        
        from .data_io import standardise_points
        pts_norm, means, stds = standardise_points(filtered_pts)
        x_norm, y_norm, z_norm = pts_norm[:, 0], pts_norm[:, 1], pts_norm[:, 2]
        
        θt = np.array(model["theta_top"])
        θb = np.array(model["theta_bottom"])
        
        plot_3d_points_by_z(
            x, y, z,
            f"Original Point Cloud - {base_name}",
            f"{base_name}_01_original_by_z",
            plots_dir
        )
        
        plot_3d_with_surfaces(
            x_norm, y_norm, z_norm, labels_filtered, θt, θb, means, stds,
            f"Fitted Surfaces - {base_name}",
            f"{base_name}_02_with_surfaces",
            plots_dir
        )
        
        plot_3d_by_distance_to_manifolds(
            x_norm, y_norm, z_norm, labels_filtered, θt, θb, means, stds,
            f"Distance to Nearest Surface - {base_name}",
            f"{base_name}_03_by_distance",
            plots_dir
        )
        
        plot_3d_by_classification(
            x_norm, y_norm, z_norm, labels_filtered,
            f"Surface Classification - {base_name}",
            f"{base_name}_04_by_classification",
            plots_dir
        )
        
        if combined_dashboard:
            plot_combined_dashboard(
                x_norm, y_norm, z_norm, labels_filtered, θt, θb, means, stds,
                f"Combined Dashboard - {base_name}",
                f"{base_name}_combined_dashboard",
                plots_dir
            )
            
    except ImportError as e:
        warnings.warn(f"Could not generate plots: {e}. Install plotly for plotting functionality.")
    except Exception as e:
        warnings.warn(f"Error generating plots for {base_name}: {e}")
