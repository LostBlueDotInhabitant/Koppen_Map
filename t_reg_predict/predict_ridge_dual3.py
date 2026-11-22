"""Predict the local temperature of every 1-degree pixel for one year (or 21ka).

Run:  python predict_ridge.py 2050 --scenario ssp245
      python predict_ridge.py --epoch 21ka
      python predict_ridge.py            (asks for the year)
Needs: kernelize_predict.py (or kernelize.py) and trained ocean/land model files + metadata JSON.
"""
import argparse
import json

import joblib
import numpy as np

import kernelize_exp as kz

# ----------------------------- settings (edit these) -----------------------------
DATA_DIRECTORY = "data_cache"   # folder containing MapVariables.csv, co2_temperature_*.csv
GITHUB_BASE_URL = None

OCEAN_MODEL_FILE = "ridge_temperature_model_sea.joblib"
LAND_MODEL_FILE = "ridge_temperature_model.joblib"

METADATA_FILE = "ridge_temperature_model_meta.json"
OCEAN_METADATA_FILE = None      # optional: set if ocean model has its own metadata JSON
LAND_METADATA_FILE = None       # optional: set if land model has its own metadata JSON

SAVE_PREDICTION_ARRAY = True    # also write predictions_<label>_<year>.npy (or predictions_21ka.npy), shape (180, 360), in C


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("year", nargs="?", type=int, help="year between 1900 and 2099 (ignored when --epoch 21ka)")
    parser.add_argument("--scenario", default="ssp245", choices=kz.ALL_SCENARIOS,
                        help="SSP scenario (only matters for years >= 2015)")
    parser.add_argument("--epoch", default="0ka", choices=("0ka", "21ka"),
                        help="map/climate epoch to predict (default: 0ka)")
    args = parser.parse_args()
    epoch = args.epoch

    if epoch == "0ka":
        year = args.year if args.year is not None else int(input("Year to predict (1900-2099): "))
        if not 1900 <= year <= 2099:
            raise SystemExit("Year must be between 1900 and 2099.")
        scenario = args.scenario
        label = "historical" if year <= 2014 else scenario
    else:
        year = None
        scenario = None
        label = "21ka"

    ocean_meta = _load_json(OCEAN_METADATA_FILE or METADATA_FILE)
    land_meta = _load_json(LAND_METADATA_FILE or METADATA_FILE)

    # Use ocean_meta for kernel/data setup (or land_meta if they differ)
    meta = ocean_meta

    ocean_model = joblib.load(OCEAN_MODEL_FILE)
    land_model = joblib.load(LAND_MODEL_FILE)

    kz.configure_data_sources(github_base_url=GITHUB_BASE_URL, temperature_directory=DATA_DIRECTORY)
    kz.configure_kernel(kernel_size=meta["kernel_size"],
                        water_vector_kernel_size=meta["water_vector_kernel_size"],
                        water_inverse_distance_power=meta["water_inverse_distance_power"])

    print(f"Loading data for {label}...")
    map_variables = kz.load_time_invariant_map_variables(epoch)
    land_fraction_flat = map_variables[..., 2].ravel()
    is_land = land_fraction_flat > 0
    is_ocean = ~is_land

    if epoch == "0ka":
        temperature_by_year, co2_by_year = kz.load_global_mean_temperature_and_co2_by_year(scenario, year)
        global_temp = temperature_by_year[year]
        co2_ppm = co2_by_year[year]
    else:
        global_temp, co2_ppm = kz.load_21ka_global_mean_temperature_and_co2()

    X = kz.make_temperature_regression_features(
        meta["kernel_size"], global_temp, co2_ppm, epoch=epoch, map_variables=map_variables
    )
    if X.shape[1] != ocean_meta["n_features"] or X.shape[1] != land_meta["n_features"]:
        raise SystemExit(
            f"Feature count {X.shape[1]} does not match model metadata "
            f"(ocean: {ocean_meta['n_features']}, land: {land_meta['n_features']}); "
            "retrain with the current kernelize module."
        )

    flat_predictions = np.empty(X.shape[0], dtype=np.float64)
    if np.any(is_ocean):
        flat_predictions[is_ocean] = ocean_model.predict(X[is_ocean])
    if np.any(is_land):
        flat_predictions[is_land] = land_model.predict(X[is_land])

    predicted_grid = kz.reshape_predictions_to_grid(flat_predictions)                      # (180, 360), C
    if SAVE_PREDICTION_ARRAY:
        out_name = f"predictions_{label}_{year}.npy" if epoch == "0ka" else "predictions_21ka.npy"
        np.save(out_name, predicted_grid)
        print(f"Saved prediction array to {out_name}")

    target_label = f"{year} ({label})" if epoch == "0ka" else "21ka"
    print(f"\nResults for {target_label} over {predicted_grid.size:,} pixels ({is_land.sum():,} land, {is_ocean.sum():,} ocean)")
    print(f"  Predicted global mean:     {predicted_grid.mean():.2f} C (input temp_C={global_temp}, co2_ppm={co2_ppm})")
    if np.any(is_land):
        print(f"  Predicted land mean:       {predicted_grid.ravel()[is_land].mean():.2f} C")
    if np.any(is_ocean):
        print(f"  Predicted ocean mean:      {predicted_grid.ravel()[is_ocean].mean():.2f} C")


if __name__ == "__main__":
    main()
