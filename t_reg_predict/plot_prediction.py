import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import kernelize_exp as kz

NPY_FILE = "predictions_historical_1990.npy"         # edit this (file made by predict_ridge.py, same folder)
DATA_DIRECTORY = "data_cache"           # folder containing MapVariables.csv, co2_temperature_*.csv
GITHUB_BASE_URL = None                  # optional: set if MapVariables.csv is not yet cached in DATA_DIRECTORY

# Optional comparison grid for 21ka (shape (180, 360), row 0 = south, e.g. a reference or baseline grid)
REFERENCE_21KA_NPY = None               # e.g. "lgm_reference_21ka.npy"; if None, compares 21ka vs 0ka map variables

# colours: 0 C is light blue (the middle of the scale); colder -> deeper blue/purple, warmer -> green/yellow/red
cmap = LinearSegmentedColormap.from_list("zero_light_blue", [
    (0.00, "#2b1055"), (0.25, "#2c4fbf"), (0.50, "#a6dcf5"),   # light blue at the centre = 0 C
    (0.65, "#7ccf6b"), (0.80, "#ffe14d"), (0.92, "#f58a1f"), (1.00, "#b3001b")])

diff_cmap = "RdBu_r"

predicted = np.load(NPY_FILE)                                                   # (180, 360), row 0 = south
kz.configure_data_sources(github_base_url=GITHUB_BASE_URL, temperature_directory=DATA_DIRECTORY)

match_21ka = re.fullmatch(r"predictions_21ka(?:_.*)?\.npy", NPY_FILE)
if match_21ka:
    if REFERENCE_21KA_NPY is not None:
        ref_grid = np.load(REFERENCE_21KA_NPY)
        ref_label = f"Reference ({REFERENCE_21KA_NPY})"
    else:
        # Compare 21ka predicted temperatures against a baseline derived from 0ka global temp & altitude
        map_0ka = kz.load_time_invariant_map_variables("0ka")
        temp_by_year_0ka, _ = kz.load_global_mean_temperature_and_co2_by_year("ssp245", 2000)
        base_global_0ka = temp_by_year_0ka[2000]
        alt_0ka = map_0ka[..., 0]
        # Simple lapse-rate baseline (-6.5 C/km relative to global mean) without needing local temp CSVs
        ref_grid = base_global_0ka - 0.0065 * np.maximum(alt_0ka, 0.0)
        ref_label = "0ka Lapse Baseline (2000)"

    diff_grid = predicted - ref_grid
    max_abs_diff = max(float(np.abs(diff_grid).max()), 0.1)

    low = min(float(predicted.min()), float(ref_grid.min()), -0.1)
    high = max(float(predicted.max()), float(ref_grid.max()), 0.1)
    norm = TwoSlopeNorm(vmin=low, vcenter=0, vmax=high)
    diff_norm = TwoSlopeNorm(vmin=-max_abs_diff, vcenter=0, vmax=max_abs_diff)

    fig, axes = plt.subplots(3, 1, figsize=(11, 14), sharex=True)

    im0 = axes[0].imshow(ref_grid, origin="lower", extent=[-180, 180, -90, 90], cmap=cmap, norm=norm)
    axes[0].set_title(f"{ref_label} (°C)")
    axes[0].set_ylabel("Latitude (°)")
    fig.colorbar(im0, ax=axes[0], label="Temperature (°C)", shrink=0.85)

    im1 = axes[1].imshow(predicted, origin="lower", extent=[-180, 180, -90, 90], cmap=cmap, norm=norm)
    axes[1].set_title("Predicted, 21ka (°C)")
    axes[1].set_ylabel("Latitude (°)")
    fig.colorbar(im1, ax=axes[1], label="Temperature (°C)", shrink=0.85)

    im2 = axes[2].imshow(diff_grid, origin="lower", extent=[-180, 180, -90, 90], cmap=diff_cmap, norm=diff_norm)
    axes[2].set_title(f"Difference: Predicted 21ka minus {ref_label} (°C)")
    axes[2].set_xlabel("Longitude (°)")
    axes[2].set_ylabel("Latitude (°)")
    fig.colorbar(im2, ax=axes[2], label="Δ Temperature (°C)", shrink=0.85)
else:
    # year and scenario are read from the file name, e.g. predictions_ssp245_2050.npy / predictions_historical_2000.npy
    matched = re.fullmatch(r"predictions_(\w+?)_(\d{4})\.npy", NPY_FILE)
    if not matched:
        raise ValueError(f"Unrecognized prediction filename format: {NPY_FILE}")
    scenario_name, year = matched.groups()
    year = int(year)
    scenario = "ssp245" if scenario_name == "historical" else scenario_name

    temp_by_year, co2_by_year = kz.load_global_mean_temperature_and_co2_by_year(scenario, year)
    map_0ka = kz.load_time_invariant_map_variables("0ka")
    alt_0ka = map_0ka[..., 0]
    # Without local ground-truth CSVs, compare predicted grid against global mean temperature
    ref_grid = np.full_like(predicted, temp_by_year[year], dtype=np.float64)
    diff_grid = predicted - ref_grid
    max_abs_diff = max(float(np.abs(diff_grid).max()), 0.1)

    low = min(float(predicted.min()), float(ref_grid.min()), -0.1)
    high = max(float(predicted.max()), float(ref_grid.max()), 0.1)
    norm = TwoSlopeNorm(vmin=low, vcenter=0, vmax=high)
    diff_norm = TwoSlopeNorm(vmin=-max_abs_diff, vcenter=0, vmax=max_abs_diff)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    im0 = axes[0].imshow(predicted, origin="lower", extent=[-180, 180, -90, 90], cmap=cmap, norm=norm)
    axes[0].set_title(f"Predicted, {year} ({scenario_name})")
    axes[0].set_ylabel("Latitude (°)")
    fig.colorbar(im0, ax=axes[0], label="Temperature (°C)", shrink=0.85)

    im1 = axes[1].imshow(diff_grid, origin="lower", extent=[-180, 180, -90, 90], cmap=diff_cmap, norm=diff_norm)
    axes[1].set_title(f"Anomaly vs. Global Mean ({temp_by_year[year]:.2f} °C), {year} ({scenario_name})")
    axes[1].set_xlabel("Longitude (°)")
    axes[1].set_ylabel("Latitude (°)")
    fig.colorbar(im1, ax=axes[1], label="Δ Temperature (°C)", shrink=0.85)

plt.tight_layout()
plt.show()
