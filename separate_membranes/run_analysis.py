#!/usr/bin/env python3
"""
Batch runner for membrane separation analysis on CSV files.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime

from .data_io import load_point_cloud
from .alternating_fit import alternating_surface_fit
from .visualise import (
    plot_3d_points_by_z,
    plot_3d_with_surfaces,
    plot_3d_by_distance_to_manifolds,
    plot_3d_by_classification,
    plot_combined_dashboard
)


def create_plot_directory(base_dir="plots"):
    """Create a timestamped directory for plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = Path(base_dir) / timestamp
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def main():
    parser = argparse.ArgumentParser(description="Batch membrane separation on CSVs.")
    parser.add_argument("--data-dir", required=True, help="Directory containing CSV files (must contain columns mx, my, mz).")
    parser.add_argument("--results-dir", default="results", help="Output directory for labeled CSVs and summary statistics.")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum number of attempts to optimise surface fitting.")
    parser.add_argument("--tol", type=float, default=0.01, help="Convergence tolerance for surface fitting.")
    parser.add_argument("--alpha", type=float, default=1e-3, help="Regularisation parameter for ridge regression.")
    parser.add_argument("--soft-gap", action="store_true", default=True, help="Use soft gap penalty instead of hard constraint (default: True).")
    parser.add_argument("--lambda-gap", type=float, default=100.0, help="Weight for soft gap penalty (default: 100.0).")
    parser.add_argument("--delta", type=float, default=None, help="Override normalized gap; default computed from data.")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots.")
    parser.add_argument("--combined-dashboard", action="store_true", help="Generate combined dashboard with all plots in one view.")
    parser.add_argument("--flat-surfaces", action="store_true", help="Use stronger regularization to create flatter surfaces (reduces edge artifacts).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    base_out_dir = Path(args.results_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_dir = None
    if not args.no_plots:
        plot_dir = create_plot_directory()
        print(f"Plots will be saved to: {plot_dir}")
    
    print(f"Results will be saved to: {out_dir}")
    
    use_soft_gap = args.soft_gap

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {data_dir}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {data_dir}")
    
    for fpath in csv_files:
        print(f"\nProcessing: {fpath.name}")
        try:
            pts = load_point_cloud(str(fpath))
            x, y, z = pts[:,0], pts[:,1], pts[:,2]

            if plot_dir is not None:
                print(f"  Generating plots for {fpath.stem}...")

            result = alternating_surface_fit(
                x, y, z,
                Delta=args.delta, alpha=args.alpha,
                max_iter=args.max_iter, tol=args.tol,
                soft_gap=use_soft_gap, lambda_gap=args.lambda_gap,
                flat_surfaces=args.flat_surfaces
            )

            df = pd.read_csv(fpath)
            labels_full = result["labels_full"]
            label_map = {1: "upper", 0: "lower", -1: "outlier"}
            df["membrane"] = [label_map[int(v)] for v in labels_full]

            out_csv = out_dir / (fpath.stem + "_labeled.csv")
            df.to_csv(out_csv, index=False)

            model = result["model"]
            metrics = result["metrics"]
            summary = {"file": fpath.name, "model": model, "metrics": metrics}
            out_json = out_dir / (fpath.stem + "_metrics.json")
            with open(out_json, "w") as fp:
                json.dump(summary, fp, indent=2)

            np.save(out_dir / (fpath.stem + "_theta_top.npy"), np.array(model["theta_top"]))
            np.save(out_dir / (fpath.stem + "_theta_bottom.npy"), np.array(model["theta_bottom"]))

            if plot_dir is not None:
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
                    f"Original Point Cloud - {fpath.stem}",
                    f"{fpath.stem}_01_original_by_z",
                    plot_dir
                )
                
                plot_3d_with_surfaces(
                    x_norm, y_norm, z_norm, labels_filtered, θt, θb, means, stds,
                    f"Fitted Surfaces - {fpath.stem}",
                    f"{fpath.stem}_02_with_surfaces",
                    plot_dir
                )
                
                plot_3d_by_distance_to_manifolds(
                    x_norm, y_norm, z_norm, labels_filtered, θt, θb, means, stds,
                    f"Distance to Nearest Surface - {fpath.stem}",
                    f"{fpath.stem}_03_by_distance",
                    plot_dir
                )
                
                plot_3d_by_classification(
                    x_norm, y_norm, z_norm, labels_filtered,
                    f"Surface Classification - {fpath.stem}",
                    f"{fpath.stem}_04_by_classification",
                    plot_dir
                )
                
                if args.combined_dashboard:
                    print(f"  Generating combined dashboard for {fpath.stem}...")
                    plot_combined_dashboard(
                        x_norm, y_norm, z_norm, labels_filtered, θt, θb, means, stds,
                        f"Combined Dashboard - {fpath.stem}",
                        f"{fpath.stem}_combined_dashboard",
                        plot_dir
                    )

            print(f"Saved: {out_csv.name}, {out_json.name}")
            print(f"  Points: {len(df)} total, {np.sum(labels_full >= 0)} kept, {np.sum(labels_full == -1)} outliers")
            print(f"  Upper: {np.sum(labels_full == 1)}, Lower: {np.sum(labels_full == 0)}")
            print(f"  RMSE (norm): top={metrics['RMSE_top_norm']:.4f}, bottom={metrics['RMSE_bottom_norm']:.4f}")
            print(f"  Min gap (norm): {metrics['achieved_min_gap_norm']:.4f}")
            if plot_dir is not None:
                print(f"  Plots saved to: {plot_dir}")
            
        except Exception as e:
            print(f"ERROR processing {fpath.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
