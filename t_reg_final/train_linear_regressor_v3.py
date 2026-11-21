"""Training pipeline for local mean temperature using Linear/Ridge regression and kernelize_2."""

import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import kernelize as kz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GITHUB_BASE_URL = "https://raw.githubusercontent.com/LostBlueDotInhabitant/Koppen_Map/refs/heads/main/"
TEMPERATURE_DIR = "TemperaturePerLocation"
CACHE_DIR = "data_cache"

KERNEL_SIZE = 5
WATER_VECTOR_KERNEL_SIZE = 9
INCLUDE_WATER_VECTORS = True
INVERSE_DISTANCE_POWER = 1
WATER_INVERSE_DISTANCE_POWER = 2
SCALE_EW_BY_COS_LAT = False

SCENARIOS = kz.ALL_SCENARIOS
YEARS = range(1900, 2100)
PIXELS_PER_YEAR = 5000
TEST_FRACTION = 0.2
SPLIT_SEED = 0
MODEL_OUTPUT_PATH = "temp_linear_regressor_k5_w9.joblib"


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

    X, y, year_of_row, scenario_of_row = kz.make_temperature_regression_training_data_from_files(
        kernel_size=None,
        scenarios=SCENARIOS,
        years=YEARS,
        pixels_per_year=PIXELS_PER_YEAR,
        global_temperature_from="file",
    )

    X_train, y_train, X_test, y_test, year_train, year_test, scen_train, scen_test = (
        kz.random_train_test_split(
            X, y, TEST_FRACTION, SPLIT_SEED, year_of_row, scenario_of_row
        )
    )

    return X_train, y_train, X_test, y_test, year_test, scen_test


def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge (alpha=10.0)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }

    best_name = None
    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, test_preds)
        rmse = root_mean_squared_error(y_test, test_preds)
        print(f"{name:20s} -> Test MAE: {mae:.4f} C | Test RMSE: {rmse:.4f} C")

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    print(f"Selected best model: {best_name}")
    return best_model


def main():
    X_train, y_train, X_test, y_test, _, _ = build_datasets()
    model = train_and_evaluate(X_train, y_train, X_test, y_test)

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
        },
        MODEL_OUTPUT_PATH,
    )
    print(f"Saved trained model to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
