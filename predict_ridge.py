"""Predict the local temperature of every 1-degree pixel for one year and compare with the real values.

Altitude handling: before the kernels are built, the altitude map is re-weighted pixel by pixel:
LAND pixels (land fraction > LAND_FRACTION_THRESHOLD %) get their altitude multiplied by LAND_ALTITUDE_WEIGHT,
OCEAN pixels (seabed depth) by OCEAN_ALTITUDE_WEIGHT (almost nothing). Because this is done on the map, every
neighbour inside a kernel also gets its own land/ocean weight, so coastlines behave. kernelize.py is NOT modified.

Run:  python predict_ridge.py 2050 --scenario ssp245
      python predict_ridge.py            (asks for the year)
Needs: kernelize.py, ridge_temperature_model.joblib and ridge_temperature_model_meta.json (from train_ridge.py).
"""
import argparse
import json

import joblib
import numpy as np

import kernelize as kz

# ----------------------------- settings (edit these) -----------------------------
DATA_DIRECTORY = "data_cache"   # same folder as in train_ridge.py
GITHUB_BASE_URL = None          # same value as in train_ridge.py (needed to load MapVariables.csv)

MODEL_FILE = "ridge_temperature_model.joblib"
METADATA_FILE = "ridge_temperature_model_meta.json"
SAVE_PREDICTION_ARRAY = True      # also write predictions_<scenario>_<year>.npy, shape (180, 360), in C
SAVE_PLOT = True                  # also write comparison_<scenario>_<year>.png (Actual on top, Predicted below)

# How much altitude counts. 1.0 = exactly as trained.
LAND_ALTITUDE_WEIGHT = 3.0        # land: mountains matter a lot (raise it for stronger mountains)
OCEAN_ALTITUDE_WEIGHT = 0.02      # ocean: seabed depth is almost ignored
LAND_FRACTION_THRESHOLD = 50.0    # a pixel is land if its land area fraction (%) is above this
# ---------------------------------------------------------------------------------


def reweight_altitude_by_surface_type(map_variables):
    """map_variables (180, 360, 4) = [altitude, ice thickness, land fraction %, ice fraction %].
    Returns (copy with re-weighted altitude, land mask (180, 360) bool)."""
    land = map_variables[..., 2] > LAND_FRACTION_THRESHOLD
    adjusted = np.array(map_variables, dtype=np.float32, copy=True)
    adjusted[..., 0] *= np.where(land, LAND_ALTITUDE_WEIGHT, OCEAN_ALTITUDE_WEIGHT).astype(np.float32)
    return adjusted, land


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("year", nargs="?", type=int, help="year between 1900 and 2099")
    parser.add_argument("--scenario", default="ssp245", choices=kz.ALL_SCENARIOS,
                        help="SSP scenario (only matters for years >= 2015)")
    args = parser.parse_args()
    year = args.year if args.year is not None else int(input("Year to predict (1900-2099): "))
    if not 1900 <= year <= 2099:
        raise SystemExit("Year must be between 1900 and 2099.")
    scenario = args.scenario
    label = "historical" if year <= 2014 else scenario

    with open(METADATA_FILE) as f:
        meta = json.load(f)
    model = joblib.load(MODEL_FILE)
    kz.configure_data_sources(github_base_url=GITHUB_BASE_URL, temperature_directory=DATA_DIRECTORY)
    kz.configure_kernel(kernel_size=meta["kernel_size"],
                        water_vector_kernel_size=meta["water_vector_kernel_size"],
                        water_inverse_distance_power=meta["water_inverse_distance_power"])

    print(f"Loading data for {year} ({label})...")
    real_grid = kz.load_local_temperature_grids_by_year(scenario, year, year)[year]        # (180, 360), C
    temperature_by_year, co2_by_year = kz.load_global_mean_temperature_and_co2_by_year(scenario, year)

    map_variables, land_grid = reweight_altitude_by_surface_type(kz.load_time_invariant_map_variables("0ka"))
    print(f"Altitude weights: land x{LAND_ALTITUDE_WEIGHT}, ocean x{OCEAN_ALTITUDE_WEIGHT} "
          f"({land_grid.mean() * 100:.1f}% of pixels are land)")

    X = kz.make_temperature_regression_features(meta["kernel_size"], temperature_by_year[year], co2_by_year[year],
                                                map_variables=map_variables)
    if X.shape[1] != meta["n_features"]:
        raise SystemExit(f"Feature count {X.shape[1]} != {meta['n_features']} the model was trained with; "
                         "retrain with the current kernelize.py.")
    predicted_grid = kz.reshape_predictions_to_grid(model.predict(X))                       # (180, 360), C
    if SAVE_PREDICTION_ARRAY:
        out_name = f"predictions_{label}_{year}.npy"
        np.save(out_name, predicted_grid)
        print(f"Saved prediction array to {out_name}")

    if SAVE_PLOT:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vmin = float(min(real_grid.min(), predicted_grid.min()))
        vmax = float(max(real_grid.max(), predicted_grid.max()))
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
        for ax, grid, title in ((axes[0], real_grid, "Actual"), (axes[1], predicted_grid, "Predicted")):
            im = ax.imshow(grid, origin="lower", extent=(-180, 180, -90, 90), cmap="turbo",   # row 0 = south
                           vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"{title}, {year} ({label})")
            ax.set_ylabel("Latitude (°)")
        axes[1].set_xlabel("Longitude (°)")
        fig.colorbar(im, ax=axes, shrink=0.8, label="Temperature (°C)")
        plot_name = f"comparison_{label}_{year}.png"
        fig.savefig(plot_name, dpi=150)
        plt.close(fig)
        print(f"Saved comparison plot to {plot_name}")

    land = land_grid.ravel()
    error_c = (predicted_grid - real_grid).ravel().astype(np.float64)                       # signed error in C
    # Percent error is taken in KELVIN: in Celsius, pixels near 0 C would divide by ~0 and blow up.
    real_k = real_grid.ravel().astype(np.float64) + 273.15
    error_pct = 100.0 * error_c / real_k

    def report(title, mask):
        e, p = error_c[mask], error_pct[mask]
        print(f"  {title:<6} MAE {np.abs(e).mean():.3f} C | bias {e.mean():+.3f} C | RMSE {np.sqrt((e ** 2).mean()):.3f} C"
              f" | MAE {np.abs(p).mean():.4f} %")

    print(f"\nResults for {year} over {error_c.size:,} pixels")
    print(f"  Mean error (bias):         {error_c.mean():+.3f} C   |  {error_pct.mean():+.4f} %")
    print(f"  Std dev of error:          {error_c.std():.3f} C   |  {error_pct.std():.4f} %")
    print(f"  Mean absolute error:       {np.abs(error_c).mean():.3f} C   |  {np.abs(error_pct).mean():.4f} %")
    print(f"  RMSE:                      {np.sqrt((error_c ** 2).mean()):.3f} C")
    print(f"  Global mean  real: {real_grid.mean():.2f} C   predicted: {predicted_grid.mean():.2f} C")
    print("\nBy surface type:")
    report("Land", land)
    report("Ocean", ~land)

    if meta.get("split_mode") == "years":
        held_out = year % meta["holdout_every"] == 0
        print(f"\nThis year was {'HELD OUT of training (honest test)' if held_out else 'part of the training data (optimistic)'}.")
    else:
        print("\nNote: training used a random row split, so this year's pixels were probably partly in the "
              "training data; these numbers are optimistic.")


if __name__ == "__main__":
    main()
