"""Training and inference pipeline for local mean temperature regression using kernelize.py."""

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import kernelize as kz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GITHUB_BASE_URL = "https://raw.githubusercontent.com/<user>/<repo>/<branch>/<folder>/"
TEMPERATURE_DIR = "C:/path/to/your/temperature/csvs"
CACHE_DIR = "data_cache"

KERNEL_SIZE = 11
WATER_VECTOR_KERNEL_SIZE = None  # None -> uses KERNEL_SIZE (or set e.g. 21 to look further for water)
INCLUDE_WATER_VECTORS = True
INVERSE_DISTANCE_POWER = 1
WATER_INVERSE_DISTANCE_POWER = 2
SCALE_EW_BY_COS_LAT = False

SCENARIO = "ssp245"
TRAIN_YEARS = list(range(1900, 2060))
VAL_YEARS = list(range(2060, 2100))
PIXELS_PER_YEAR = 5000
MODEL_OUTPUT_PATH = "temp_regressor_11x11.joblib"


def configure_pipeline():
    kz.configure_data_sources(
        github_base_url=GITHUB_BASE_URL,
        temperature_directory=TEMPERATURE_DIR,
        cache_directory=CACHE_DIR,
    )
    kz.configure_kernel(
        kernel_size=KERNEL_SIZE,
        water_vector_kernel_size=WATER_VECTOR_KERNEL_SIZE,
        include_water_vectors=INCLUDE_WATER_VECTORS,
        inverse_distance_power=INVERSE_DISTANCE_POWER,
        water_inverse_distance_power=WATER_INVERSE_DISTANCE_POWER,
        scale_east_west_by_cos_latitude=SCALE_EW_BY_COS_LAT,
    )


def build_datasets():
    configure_pipeline()

    # Pass kernel_size=None so kernelize uses the size set in configure_kernel()
    X_train, y_train = kz.make_temperature_regression_training_data_from_files(
        kernel_size=None,
        scenario=SCENARIO,
        years=TRAIN_YEARS,
        pixels_per_year=PIXELS_PER_YEAR,
        global_temperature_from="file",
    )

    X_val, y_val = kz.make_temperature_regression_training_data_from_files(
        kernel_size=None,
        scenario=SCENARIO,
        years=VAL_YEARS,
        pixels_per_year=PIXELS_PER_YEAR,
        global_temperature_from="file",
    )

    return X_train, y_train, X_val, y_val


def train_and_evaluate(X_train, y_train, X_val, y_val):
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = root_mean_squared_error(y_val, val_preds)

    print(f"Validation MAE:  {mae:.4f} C")
    print(f"Validation RMSE: {rmse:.4f} C")

    return model


def predict_global_grid_for_year(model, target_year, scenario=SCENARIO, epoch="0ka"):
    global_temp_by_year, co2_by_year = kz.load_global_mean_temperature_and_co2_by_year(
        scenario=scenario,
        last_year=target_year,
    )

    # Pass kernel_size=None so kernelize uses the configured kernel & water-vector settings
    X_grid = kz.make_temperature_regression_features(
        kernel_size=None,
        global_mean_temperature=global_temp_by_year[target_year],
        co2_ppm=co2_by_year[target_year],
        epoch=epoch,
    )

    flat_preds = model.predict(X_grid)
    return kz.reshape_predictions_to_grid(flat_preds)


def main():
    X_train, y_train, X_val, y_val = build_datasets()
    model = train_and_evaluate(X_train, y_train, X_val, y_val)

    joblib.dump(
        {
            "model": model,
            "kernel_settings": {
                "kernel_size": KERNEL_SIZE,
                "water_vector_kernel_size": WATER_VECTOR_KERNEL_SIZE,
                "include_water_vectors": INCLUDE_WATER_VECTORS,
                "inverse_distance_power": INVERSE_DISTANCE_POWER,
                "water_inverse_distance_power": WATER_INVERSE_DISTANCE_POWER,
                "scale_east_west_by_cos_latitude": SCALE_EW_BY_COS_LAT,
            },
            "scenario": SCENARIO,
        },
        MODEL_OUTPUT_PATH,
    )
    print(f"Saved trained model to {MODEL_OUTPUT_PATH}")

    sample_year = 2075
    grid_2075 = predict_global_grid_for_year(model, target_year=sample_year)
    print(f"Predicted {sample_year} global grid shape: {grid_2075.shape}")
    print(f"Min/Max predicted temperature in {sample_year}: {grid_2075.min():.2f} C / {grid_2075.max():.2f} C")


if __name__ == "__main__":
    main()
