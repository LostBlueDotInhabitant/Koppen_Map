import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import kernelize as kz

NPY_FILE = "predictions_21ka.npy"         # edit this (file made by predict_ridge.py, same folder)
DATA_DIRECTORY = "data_cache"   # edit this (same folder as in the other scripts)

# colours: 0 C is light blue (the middle of the scale); colder -> deeper blue/purple, warmer -> green/yellow/red
cmap = LinearSegmentedColormap.from_list("zero_light_blue", [
    (0.00, "#2b1055"), (0.25, "#2c4fbf"), (0.50, "#a6dcf5"),   # light blue at the centre = 0 C
    (0.65, "#7ccf6b"), (0.80, "#ffe14d"), (0.92, "#f58a1f"), (1.00, "#b3001b")])

predicted = np.load(NPY_FILE)                                                   # (180, 360), row 0 = south

match_21ka = re.fullmatch(r"predictions_21ka(?:_.*)?\.npy", NPY_FILE)
if match_21ka:
    low = min(predicted.min(), -0.1)
    high = max(predicted.max(), 0.1)
    norm = TwoSlopeNorm(vmin=low, vcenter=0, vmax=high)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(predicted, origin="lower", extent=[-180, 180, -90, 90], cmap=cmap, norm=norm)
    ax.set_title("Predicted, 21ka")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    fig.colorbar(image, ax=ax, label="Temperature (°C)", shrink=0.85)
else:
    # year and scenario are read from the file name, e.g. predictions_ssp245_2050.npy / predictions_historical_2000.npy
    matched = re.fullmatch(r"predictions_(\w+?)_(\d{4})\.npy", NPY_FILE)
    if not matched:
        raise ValueError(f"Unrecognized prediction filename format: {NPY_FILE}")
    scenario_name, year = matched.groups()
    year = int(year)
    scenario = "ssp245" if scenario_name == "historical" else scenario_name

    kz.configure_data_sources(temperature_directory=DATA_DIRECTORY)
    actual = kz.load_local_temperature_grids_by_year(scenario, year, year)[year]    # (180, 360)

    low = min(predicted.min(), actual.min(), -0.1)
    high = max(predicted.max(), actual.max(), 0.1)
    norm = TwoSlopeNorm(vmin=low, vcenter=0, vmax=high)                              # same scale for both maps

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    for ax, grid, title in [(axes[0], actual, f"Actual, {year} ({scenario_name})"),
                            (axes[1], predicted, f"Predicted, {year} ({scenario_name})")]:
        image = ax.imshow(grid, origin="lower", extent=[-180, 180, -90, 90], cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_ylabel("Latitude (°)")
    axes[1].set_xlabel("Longitude (°)")
    fig.colorbar(image, ax=axes, label="Temperature (°C)", shrink=0.8)

plt.tight_layout()
plt.show()
