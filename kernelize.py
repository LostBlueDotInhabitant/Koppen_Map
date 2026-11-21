"""Neighbourhood (kernel) features for the Koppen decision tree and the local-temperature regression.

Grid convention everywhere: shape (180, 360); row = latitude (south -> north),
column = longitude (-180 -> 180, west -> east). Longitude wraps around; crossing a pole lands on the
same latitude row at longitude + 180 degrees. Nothing is loaded at import time.
"""
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DATA_DIR = "/mnt/user-data/uploads/"
NUM_LAT_ROWS, NUM_LON_COLS = 180, 360


# ======================= grid conversion =======================
def latlon_to_grid_indices(latitudes, longitudes):
    """Latitudes/longitudes of 1-degree pixel centres (longitude in either 0..360 or -180..180)
    -> (row_indices, col_indices) into a (180, 360) grid."""
    longitudes = ((np.asarray(longitudes) + 180) % 360) - 180
    return (np.round(np.asarray(latitudes) + 89.5).astype(int),
            np.round(longitudes + 179.5).astype(int))

def convert_latlon_table_to_grid(table, value_column, latitude_column="lat", longitude_column="lon",
                                 require_full_coverage=True):
    """Table with lat, lon and a value column (e.g. local temperature) -> (180, 360) grid.
    If require_full_coverage is True, raises when any of the 64,800 pixels is absent from the table;
    otherwise absent pixels become NaN."""
    grid = np.full((NUM_LAT_ROWS, NUM_LON_COLS), np.nan, dtype=np.float32)
    rows, cols = latlon_to_grid_indices(table[latitude_column].to_numpy(), table[longitude_column].to_numpy())
    grid[rows, cols] = table[value_column].to_numpy()
    n_missing = int(np.isnan(grid).sum())
    if require_full_coverage and n_missing:
        raise ValueError(f"{n_missing} of {grid.size} pixels are missing from the table")
    return grid


# ======================= loading =======================
def load_time_invariant_map_variables(epoch="0ka", data_dir=DATA_DIR):
    """Altitude, ice thickness, land fraction, ice fraction for the chosen epoch ('0ka' or '21ka').
    Returns (180, 360, 4). These maps do not change with the year, hence 'time invariant'."""
    table = pd.read_csv(data_dir + "MapVariables.csv")
    names = [f"altitude_{epoch}_m", f"ice_thickness_{epoch}_m",
             f"land_area_fraction_{epoch}_pct", f"ice_area_fraction_{epoch}_pct"]
    return np.stack([convert_latlon_table_to_grid(table, n) for n in names], axis=-1)

def load_koppen_labels_for_year(year, data_dir=DATA_DIR):
    """Returns (koppen_labels, row_indices, col_indices) for every Koppen-labelled pixel in that year."""
    table = pd.read_csv(data_dir + "koppen_historical_1900_2026.csv", usecols=["lat", "lon", str(year)])
    rows, cols = latlon_to_grid_indices(table["lat"].to_numpy(), table["lon"].to_numpy())
    return table[str(year)].to_numpy(), rows, cols


# ======================= the kernel =======================
def wrap_grid_across_edges_and_poles(grid, pad_width):
    """(rows, cols, channels) -> (rows + 2*pad, cols + 2*pad, channels).
    Longitude wraps. Past a pole you see the same row, shifted 180 degrees in longitude."""
    half_turn = grid.shape[1] // 2
    rows_beyond_north = np.roll(grid[::-1][:pad_width], half_turn, axis=1)
    rows_beyond_south = np.roll(grid[:pad_width][::-1], half_turn, axis=1)
    with_poles = np.concatenate([rows_beyond_south, grid, rows_beyond_north], axis=0)
    return np.pad(with_poles, ((0, 0), (pad_width, pad_width), (0, 0)), mode="wrap")

def build_neighbourhood_features_for_every_pixel(grid, kernel_size):
    """(rows, cols, channels) -> (rows, cols, kernel_size**2 * channels).
    Each pixel gets its OWN kernel: the values of its kernel_size x kernel_size neighbourhood."""
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    padded = wrap_grid_across_edges_and_poles(grid, kernel_size // 2)
    windows = sliding_window_view(padded, (kernel_size, kernel_size), axis=(0, 1))   # (r,c,ch,k,k)
    windows = windows.transpose(0, 1, 3, 4, 2)                                       # (r,c,k,k,ch)
    return windows.reshape(grid.shape[0], grid.shape[1], -1)


# ======================= feature builders (all 64,800 pixels, for training or prediction) ==========
def make_temperature_regression_features(kernel_size, global_mean_temperature, co2_ppm, epoch="0ka",
                                         map_variables=None):
    """Features for EVERY pixel (land + ocean), shape (64800, kernel_size**2 * 4 + 2).
    Columns: kernel of [altitude, ice thickness, land fraction, ice fraction], then global temp, CO2.
    Use epoch='21ka' (and 21 ka global temp / CO2) when predicting the past."""
    if map_variables is None:
        map_variables = load_time_invariant_map_variables(epoch)
    kernels = build_neighbourhood_features_for_every_pixel(map_variables, kernel_size)
    kernels = kernels.reshape(NUM_LAT_ROWS * NUM_LON_COLS, -1)
    globals_ = np.full((len(kernels), 2), [global_mean_temperature, co2_ppm], dtype=np.float32)
    return np.hstack([kernels, globals_])

def make_koppen_tree_features(kernel_size, surface_temperature_grid, epoch="0ka", map_variables=None):
    """Features for EVERY pixel, shape (64800, kernel_size**2 * 5).
    Columns: kernel of the 4 map variables, then kernel of surface temperature (180x360 grid)."""
    if map_variables is None:
        map_variables = load_time_invariant_map_variables(epoch)
    five_channels = np.concatenate([map_variables, surface_temperature_grid[..., None].astype(np.float32)], -1)
    kernels = build_neighbourhood_features_for_every_pixel(five_channels, kernel_size)
    return kernels.reshape(NUM_LAT_ROWS * NUM_LON_COLS, -1)


# ======================= ready-to-train functions =======================
def make_temperature_regression_training_data(kernel_size, years, global_mean_temperature_by_year,
                                              co2_ppm_by_year, local_temperature_grid_by_year,
                                              map_epoch="0ka", pixels_per_year=None, random_seed=0):
    """Returns (X, y) ready for model.fit(X, y). One sample per pixel per year (all pixels, land + ocean).
    years: iterable of years. *_by_year: dicts {year: value} (grids are (180, 360)).
    pixels_per_year: randomly keep this many pixels per year to limit memory (None = all 64,800)."""
    rng = np.random.default_rng(random_seed)
    map_variables = load_time_invariant_map_variables(map_epoch)
    kernels = build_neighbourhood_features_for_every_pixel(map_variables, kernel_size)
    kernels = kernels.reshape(NUM_LAT_ROWS * NUM_LON_COLS, -1)           # same for every year: computed once
    X_parts, y_parts = [], []
    for year in years:
        keep = (np.arange(len(kernels)) if pixels_per_year is None
                else rng.choice(len(kernels), pixels_per_year, replace=False))
        globals_ = np.full((len(keep), 2),
                           [global_mean_temperature_by_year[year], co2_ppm_by_year[year]], dtype=np.float32)
        X_parts.append(np.hstack([kernels[keep], globals_]))
        y_parts.append(local_temperature_grid_by_year[year].reshape(-1)[keep])
    return np.vstack(X_parts), np.concatenate(y_parts)

def make_koppen_tree_training_data(kernel_size, koppen_year_and_surface_temperature_grid_pairs, map_epoch="0ka"):
    """Returns (X, y) ready for model.fit(X, y). One sample per Koppen-labelled pixel per pair.
    Pairs: [(koppen_label_year, surface_temperature_grid), ...], e.g. one per 30-year Koppen period
    with that period's mean temperature grid. Water pixels without a label are never centres
    but still appear inside neighbours' kernels."""
    map_variables = load_time_invariant_map_variables(map_epoch)
    X_parts, y_parts = [], []
    for koppen_year, temperature_grid in koppen_year_and_surface_temperature_grid_pairs:
        labels, rows, cols = load_koppen_labels_for_year(koppen_year)
        features = make_koppen_tree_features(kernel_size, temperature_grid, map_variables=map_variables)
        X_parts.append(features[rows * NUM_LON_COLS + cols])
        y_parts.append(labels)
    return np.vstack(X_parts), np.concatenate(y_parts)

def reshape_predictions_to_grid(flat_predictions):
    """(64800,) predictions from the all-pixel feature builders -> (180, 360) grid."""
    return np.asarray(flat_predictions).reshape(NUM_LAT_ROWS, NUM_LON_COLS)


if __name__ == "__main__":
    fake_temp = {y: np.zeros((NUM_LAT_ROWS, NUM_LON_COLS), np.float32) for y in (1900, 1901)}
    X, y = make_temperature_regression_training_data(
        5, [1900, 1901], {1900: 13.5, 1901: 13.48}, {1900: 296.4, 1901: 296.8}, fake_temp, pixels_per_year=5000)
    print("regression:", X.shape, y.shape)
    Xt, yt = make_koppen_tree_training_data(5, [(1961, fake_temp[1900]), (1991, fake_temp[1900])])
    print("tree:", Xt.shape, yt.shape)
    grid = np.arange(180 * 360, dtype=np.float32).reshape(180, 360, 1)
    print(build_neighbourhood_features_for_every_pixel(grid, 3)[179, 0].reshape(3, 3))
