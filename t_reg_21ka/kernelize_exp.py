"""Prediction-only kernel and feature module (no local temperature grid CSVs required).

Omits local temperature grid loaders (tas_annual_mean_1deg_wide_celsius.csv and
temp_annual_all_scenarios_2015_2099.csv) and training-data helpers, keeping only what is
needed to load map variables, global temp / CO2, and build prediction features.

DATA LAYOUT
  * In cache_directory (default "data_cache"), temperature_directory, or on GitHub:
        MapVariables.csv
        co2_temperature_historical_1900_2026.csv
        co2_temperature_projections_2026_2100.csv
        co2_temperature_21ka.csv

GRID CONVENTION (everywhere): array shape (180, 360); row = latitude (south -> north),
column = longitude (-180 -> 180, west -> east). Longitude wraps around the globe; going past a pole
lands on the same latitude row at longitude + 180 degrees.
"""
import os
import urllib.request
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

NUM_LAT_ROWS, NUM_LON_COLS = 180, 360          # 1-degree global grid: 180 * 360 = 64,800 pixels
NUM_PIXELS = NUM_LAT_ROWS * NUM_LON_COLS

CO2_TEMPERATURE_21KA_FILE = "co2_temperature_21ka.csv"

ALL_SCENARIOS = ("ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585")

# Where the data lives. Filled in by configure_data_sources().
_DATA_SOURCES = {"github_base_url": None, "temperature_directory": None, "cache_directory": "data_cache"}

# Kernel / vector settings. Change them from outside with configure_kernel(); any function that takes
# kernel_size accepts None, which means "use the value set here".
_KERNEL_SETTINGS = {"kernel_size": 5, "water_vector_kernel_size": None, "include_water_vectors": True,
                    "inverse_distance_power": 1, "water_inverse_distance_power": 2,
                    "scale_east_west_by_cos_latitude": False}


# ======================= kernel configuration =======================
def configure_kernel(kernel_size=5, water_vector_kernel_size=None, include_water_vectors=True,
                     inverse_distance_power=1, scale_east_west_by_cos_latitude=False,
                     water_inverse_distance_power=2):
    """Set the defaults used by every function below (each call sets ALL of them)."""
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    _KERNEL_SETTINGS.update(kernel_size=kernel_size, water_vector_kernel_size=water_vector_kernel_size,
                            include_water_vectors=include_water_vectors,
                            inverse_distance_power=inverse_distance_power,
                            scale_east_west_by_cos_latitude=scale_east_west_by_cos_latitude,
                            water_inverse_distance_power=water_inverse_distance_power)

def _resolve_kernel_size(kernel_size):
    """None -> the configured default."""
    return _KERNEL_SETTINGS["kernel_size"] if kernel_size is None else kernel_size


# ======================= data source configuration =======================
def configure_data_sources(github_base_url=None, temperature_directory=None, cache_directory="data_cache"):
    """Tell the module where data files can be found."""
    if github_base_url is not None and not github_base_url.endswith("/"):
        github_base_url += "/"
    _DATA_SOURCES.update(github_base_url=github_base_url, temperature_directory=temperature_directory,
                         cache_directory=cache_directory)

def _path_of_github_file(file_name):
    """Return local path for file_name from cache_directory or temperature_directory, or download from GitHub."""
    cached_path = os.path.join(_DATA_SOURCES["cache_directory"], file_name)
    if os.path.exists(cached_path):
        return cached_path
    if _DATA_SOURCES["temperature_directory"] is not None:
        local_path = os.path.join(_DATA_SOURCES["temperature_directory"], file_name)
        if os.path.exists(local_path):
            return local_path
    base_url = _DATA_SOURCES["github_base_url"]
    if base_url is None:
        raise RuntimeError(f"{file_name} not found in '{_DATA_SOURCES['cache_directory']}' or in the temperature "
                           "directory; put it there or call configure_data_sources(github_base_url=...).")
    os.makedirs(_DATA_SOURCES["cache_directory"], exist_ok=True)
    urllib.request.urlretrieve(base_url + file_name, cached_path)
    return cached_path


# ======================= grid conversion =======================
def latlon_to_grid_indices(latitudes, longitudes):
    """Latitudes/longitudes of 1-degree pixel centres -> (row_indices, col_indices) into (180, 360)."""
    longitudes = ((np.asarray(longitudes) + 180) % 360) - 180
    return (np.round(np.asarray(latitudes) + 89.5).astype(int),
            np.round(longitudes + 179.5).astype(int))

def convert_latlon_table_to_grid(table, value_column, latitude_column="lat", longitude_column="lon",
                                 require_full_coverage=True):
    """Table with lat, lon and a value column -> (180, 360) grid."""
    grid = np.full((NUM_LAT_ROWS, NUM_LON_COLS), np.nan, dtype=np.float32)
    rows, cols = latlon_to_grid_indices(table[latitude_column].to_numpy(), table[longitude_column].to_numpy())
    grid[rows, cols] = table[value_column].to_numpy()
    n_missing = int(np.isnan(grid).sum())
    if require_full_coverage and n_missing:
        raise ValueError(f"{n_missing} of {grid.size} pixels are missing from the table")
    return grid


# ======================= loading: map variables & global temperature / CO2 =======================
def load_time_invariant_map_variables(epoch="0ka"):
    """Altitude, ice thickness, land fraction, ice fraction for the chosen epoch ('0ka' or '21ka').
    Returns (180, 360, 4)."""
    table = pd.read_csv(_path_of_github_file("MapVariables.csv"))
    names = [f"altitude_{epoch}_m", f"ice_thickness_{epoch}_m",
             f"land_area_fraction_{epoch}_pct", f"ice_area_fraction_{epoch}_pct"]
    return np.stack([convert_latlon_table_to_grid(table, n) for n in names], axis=-1)

def load_global_mean_temperature_and_co2_by_year(scenario="ssp245", last_year=2099):
    """Returns (temperature_by_year, co2_by_year) as dicts {year: value} for 1900..last_year."""
    historical = pd.read_csv(_path_of_github_file("co2_temperature_historical_1900_2026.csv")).set_index("year")
    projections = pd.read_csv(_path_of_github_file("co2_temperature_projections_2026_2100.csv")).set_index("year")
    temperature_by_year, co2_by_year = {}, {}
    for year in range(1900, last_year + 1):
        if year <= 2025:
            temperature_by_year[year] = float(historical.loc[year, "temp_C"])
            co2_by_year[year] = float(historical.loc[year, "co2_ppm"])
        else:
            temperature_by_year[year] = float(projections.loc[year, f"temp_C_{scenario}"])
            co2_by_year[year] = float(projections.loc[year, f"co2_ppm_{scenario}"])
    return temperature_by_year, co2_by_year

def load_21ka_global_mean_temperature_and_co2():
    """Returns (temp_C, co2_ppm) as floats from co2_temperature_21ka.csv."""
    table = pd.read_csv(_path_of_github_file(CO2_TEMPERATURE_21KA_FILE))
    row = table.iloc[0]
    return float(row["temp_C"]), float(row["co2_ppm"])


# ======================= kernel & water/ice vectors =======================
def wrap_grid_across_edges_and_poles(grid, pad_width):
    """(rows, cols, channels) -> (rows + 2*pad, cols + 2*pad, channels)."""
    half_turn = grid.shape[1] // 2
    rows_beyond_north = np.roll(grid[::-1][:pad_width], half_turn, axis=1)
    rows_beyond_south = np.roll(grid[:pad_width][::-1], half_turn, axis=1)
    with_poles = np.concatenate([rows_beyond_south, grid, rows_beyond_north], axis=0)
    return np.pad(with_poles, ((0, 0), (pad_width, pad_width), (0, 0)), mode="wrap")

def build_neighbourhood_features_for_every_pixel(grid, kernel_size):
    """(rows, cols, channels) -> (rows, cols, kernel_size**2 * channels)."""
    assert kernel_size % 2 == 1, "kernel_size must be odd so the pixel sits in the middle"
    padded = wrap_grid_across_edges_and_poles(grid, kernel_size // 2)
    windows = sliding_window_view(padded, (kernel_size, kernel_size), axis=(0, 1))
    windows = windows.transpose(0, 1, 3, 4, 2)
    return windows.reshape(grid.shape[0], grid.shape[1], -1)

def compute_vectors_toward_marked_pixels(marked_pixel_grid, kernel_size=None, inverse_distance_power=None,
                                         scale_east_west_by_cos_latitude=None):
    """Returns (180, 360, 4) directional vectors toward marked pixels."""
    k = _resolve_kernel_size(_KERNEL_SETTINGS["water_vector_kernel_size"]) if kernel_size is None else kernel_size
    power = _KERNEL_SETTINGS["inverse_distance_power"] if inverse_distance_power is None else inverse_distance_power
    scale_ew = (_KERNEL_SETTINGS["scale_east_west_by_cos_latitude"] if scale_east_west_by_cos_latitude is None
                else scale_east_west_by_cos_latitude)
    assert k % 2 == 1, "vector kernel size must be odd"
    radius = k // 2
    is_marked = (marked_pixel_grid > 0).astype(np.float32)
    padded = wrap_grid_across_edges_and_poles(is_marked[..., None], radius)[..., 0]
    windows = sliding_window_view(padded, (k, k))
    offsets = np.arange(-radius, radius + 1)
    north_offset, east_offset = np.meshgrid(offsets, offsets, indexing="ij")
    cos_latitude = np.cos(np.deg2rad(np.arange(NUM_LAT_ROWS) - 89.5)) if scale_ew else np.ones(NUM_LAT_ROWS)
    east = east_offset[None, :, :] * cos_latitude[:, None, None]
    north = np.broadcast_to(north_offset[None, :, :], east.shape).astype(float)
    distance = np.sqrt(east ** 2 + north ** 2)
    is_centre = distance == 0
    safe_distance = np.where(is_centre, 1.0, distance)
    unit_east = np.where(is_centre, 0.0, east / safe_distance)
    unit_north = np.where(is_centre, 0.0, north / safe_distance)
    weight = safe_distance ** -power

    def weighted_sum(per_row_weights):
        return np.einsum("rcij,rij->rc", windows, per_row_weights.astype(np.float32))

    return np.stack([weighted_sum(unit_east), weighted_sum(unit_north),
                     weighted_sum(unit_east * weight), weighted_sum(unit_north * weight)], axis=-1)

def compute_water_vectors_for_every_pixel(land_fraction_grid, **options):
    options.setdefault("inverse_distance_power", _KERNEL_SETTINGS["water_inverse_distance_power"])
    return compute_vectors_toward_marked_pixels(land_fraction_grid == 0, **options)

def compute_open_water_vectors_for_every_pixel(land_fraction_grid, ice_fraction_grid, **options):
    options.setdefault("inverse_distance_power", _KERNEL_SETTINGS["water_inverse_distance_power"])
    return compute_vectors_toward_marked_pixels((land_fraction_grid == 0) & (ice_fraction_grid == 0), **options)

def compute_ice_vectors_for_every_pixel(ice_fraction_grid, **options):
    options.setdefault("inverse_distance_power", _KERNEL_SETTINGS["water_inverse_distance_power"])
    return compute_vectors_toward_marked_pixels(ice_fraction_grid > 0, **options)


# ======================= feature builders =======================
LOCATION_FEATURE_NAMES = ("latitude", "abs_latitude", "sin_lat", "cos_lat", "sin_lat^2", "sin_lat^3",
                          "sin_lon", "cos_lon")

def compute_location_features_for_every_pixel():
    """Where the pixel is on the globe: (64800, 8) float32 columns."""
    latitude = np.repeat(np.arange(NUM_LAT_ROWS) - 89.5, NUM_LON_COLS)
    longitude = np.tile(np.arange(NUM_LON_COLS) - 179.5, NUM_LAT_ROWS)
    sin_lat = np.sin(np.deg2rad(latitude))
    return np.stack([latitude, np.abs(latitude), sin_lat, np.cos(np.deg2rad(latitude)), sin_lat ** 2, sin_lat ** 3,
                     np.sin(np.deg2rad(longitude)), np.cos(np.deg2rad(longitude))], axis=1).astype(np.float32)

def _map_features_for_every_pixel(map_variables, kernel_size, extra_channel=None, include_location_features=False):
    kernel_size = _resolve_kernel_size(kernel_size)
    channels = map_variables if extra_channel is None else np.concatenate(
        [map_variables, extra_channel[..., None].astype(np.float32)], axis=-1)
    columns = [build_neighbourhood_features_for_every_pixel(channels, kernel_size).reshape(NUM_PIXELS, -1)]
    if _KERNEL_SETTINGS["include_water_vectors"]:
        land_fraction, ice_fraction = map_variables[..., 2], map_variables[..., 3]
        columns.append(compute_water_vectors_for_every_pixel(land_fraction).reshape(NUM_PIXELS, -1))
        columns.append(compute_open_water_vectors_for_every_pixel(land_fraction, ice_fraction)
                       .reshape(NUM_PIXELS, -1))
        columns.append(compute_ice_vectors_for_every_pixel(ice_fraction).reshape(NUM_PIXELS, -1))
    if include_location_features:
        columns.append(compute_location_features_for_every_pixel())
    return np.hstack(columns).astype(np.float32, copy=False)

def make_temperature_regression_features(kernel_size, global_mean_temperature, co2_ppm, epoch="0ka",
                                         map_variables=None):
    """Features for EVERY pixel (land + ocean) for temperature regression prediction."""
    if map_variables is None:
        map_variables = load_time_invariant_map_variables(epoch)
    kernels = _map_features_for_every_pixel(map_variables, kernel_size, include_location_features=True)
    global_columns = np.full((NUM_PIXELS, 2), [global_mean_temperature, co2_ppm], dtype=np.float32)
    return np.hstack([kernels, global_columns])

def make_koppen_tree_features(kernel_size, surface_temperature_grid, epoch="0ka", map_variables=None):
    """Features for EVERY pixel for Koppen classification prediction."""
    if map_variables is None:
        map_variables = load_time_invariant_map_variables(epoch)
    return _map_features_for_every_pixel(map_variables, kernel_size, extra_channel=surface_temperature_grid)

def reshape_predictions_to_grid(flat_predictions):
    """(64800,) predictions from all-pixel feature builders -> (180, 360) grid."""
    return np.asarray(flat_predictions).reshape(NUM_LAT_ROWS, NUM_LON_COLS)
