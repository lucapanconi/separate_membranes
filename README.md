# Membrane Separator

A Python package for estimating membrane surfaces from 3D protein maps using height-field pair fitting with cubic polynomials.

+ What are the inputs?
+ What are the outputs?
+ Advanced: changing parameters of the surface-fitting algorithm.

## Quick Start
Open your terminal and enter this to install the package:
```bash
pip install git+https://github.com/lucapanconi/membrane_separator.git
```

Create a Python script which contains the following lines:

```python
from membrane_separator import separate_membranes

# Process a directory of CSV files
separate_membranes("path/to/your/csv/files")
```

And that's it! The function will:
- Process all CSV files in the directory
- Generate labeled CSV files with membrane classifications
- Create interactive 3D visualizations
- Save all outputs to timestamped plots and results folders

## Project Structure

```
membrane_separator/
├── membrane_separator/          # Core package
│   ├── __init__.py
│   ├── separate_membranes.py    # Main user function
│   ├── data_io.py
│   ├── design_matrix.py
│   ├── qp_solver.py
│   ├── alternating_fit.py
│   ├── visualize.py
│   ├── run_analysis.py
│   ├── cli.py
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Advanced Usage

1. Clone this repository:
```bash
git clone https://github.com/lucapanconi/separate_membranes
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Implement:
```python
from membrane_separator import separate_membranes

# Process a directory of CSV files
results = separate_membranes("path/to/your/csv/files")

# Process a single DataFrame
import pandas as pd
df = pd.read_csv("your_data.csv")
results = separate_membranes(df)

# Customize parameters
results = separate_membranes(
    "path/to/data",
    results_dir="my_results", # Change default result location
    plots_dir="my_plots", # Change default plot location
    flat_surfaces=True, 
    max_iter=15,
    alpha=1e-2
)
```

You can customize the behavior of the `separate_membranes` function by supplying any of the following parameters:

| Parameter         | Type      | Description                                                                                                                                                                                        | Default    |
|-------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| `data`            | str, Path, pd.DataFrame | The data to process. Can be a string path to a directory containing CSV files, a Path object, or a pandas DataFrame.                                      | **Required** |
| `results_dir`     | str, Path | The directory to save the results.                                                                                                                          | `"results"` |
| `plots_dir`       | str, Path | The directory to save the plots.                                                                                                                            | `"plots"`   |
| `generate_plots`  | bool      | Whether to generate plots.                                                                                                                                  | `True`      |
| `combined_dashboard` | bool   | Whether to generate a combined dashboard of all plots.                                                                                                     | `True`      |
| `flat_surfaces`   | bool      | Whether to try enforcing flat surfaces. This places stronger regularisation on high-order coefficients in the surface polynomial, reducing edge artifacts.   | `True`      |
| `max_iter`        | int       | The maximum number of iterations for the alternating fit procedure.                                                                                        | `10`        |
| `tol`             | float     | The tolerance for convergence; lower means more precise but potentially slower.                                                                             | `0.01`      |
| `alpha`           | float     | Regularization parameter for the initial fit. Larger values will yield smoother surfaces.                                                                   | `1e-3`      |
| `soft_gap`        | bool      | Whether to use a soft gap constraint instead of enforcing a strict gap size.                                                                                | `False`     |
| `lambda_gap`      | float     | Weight for the soft gap constraint if enabled.                                                                                                              | `10.0`      |
| `delta`           | float     | The gap size between the two surfaces. If `None`, it will be automatically computed from the data.                                                          | `None`      |

You can pass any of these as keyword arguments to `separate_membranes()`.

## Input Data Format

CSV files must contain the following columns:
- `mx`: X-coordinates
- `my`: Y-coordinates  
- `mz`: Z-coordinates

Additional columns are preserved in the output.

## Output Files

For each input file `NAME.csv`, the following files are generated in a timestamped subfolder under the results directory:

**Results Content:**
- `NAME_labeled.csv`: Original data with added `membrane` column (`upper`, `lower`, or `outlier`)
- `NAME_metrics.json`: Model parameters, metrics, and normalization statistics
- `NAME_theta_top.npy`: Polynomial coefficients for upper surface
- `NAME_theta_bottom.npy`: Polynomial coefficients for lower surface

### Interactive 3D Visualizations

When plotting is enabled (default), interactive 3D plots are generated in a timestamped folder under `plots/`:

**Plots Content:**
- `NAME_01_original_by_z.html`: Original point cloud colored by z-height
- `NAME_02_with_surfaces.html`: Points with fitted membrane surfaces overlaid
- `NAME_03_by_distance.html`: Points colored by distance to nearest surface
- `NAME_04_by_classification.html`: Points colored by membrane classification
- `NAME_combined_dashboard.html`: (Optional) All four visualizations in one interactive 2x2 grid

All plots are interactive and can be rotated, zoomed, and panned in your web browser.

## License

See LICENSE file for details.
