"""Neighbourhood (kernel) features for the Koppen decision tree and the local-temperature regression.

DATA LAYOUT
  * On GitHub (downloaded once, then cached on disk): MapVariables.csv, koppen_historical_1900_2026.csv,
    co2_temperature_historical_1900_2026.csv, co2_temperature_projections_2026_2100.csv
  * On your computer (too big for GitHub): the two local-temperature files
        tas_annual_mean_1deg_wide_celsius.csv        (1900-2014, one column per year)
        temp_annual_all_scenarios_2015_2099.csv      (2015-2099, 7 SSP scenarios)

SETUP (call once, before anything else)
    import kernelize as kz
    kz.configure_data_sources(
        github_base_url="https://raw.githubusercontent.com/<user>/<repo>/<branch>/<folder>/",
        temperature_directory="C:/path/to/folder/with/the/two/temperature/csvs")

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

LOCAL_HISTORICAL_TEMPERATURE_FILE = "tas_annual_mean_1deg_wide_celsius.csv"
LOCAL_SCENARIO_TEMPERATURE_FILE = "temp_annual_all_scenarios_2015_2099.csv"

# Koppen labels only change every 30 years, so there are 4 distinct label maps in 1900-2026.
# Each entry: (year whose Koppen label to use, first year, last year of the period to average temperature over)
DEFAULT_KOPPEN_PERIODS = [(1900, 1900, 1930), (1931, 1931, 1960), (1961, 1961, 1990), (1991, 1991, 2020)]

# Where the data lives. Filled in by configure_data_sources().
_DATA_SOURCES = {"github_base_url": None, "temperature_directory": None, "cache_directory": "data_cache"}


# ======================= data source configuration =======================
def configure_data_sources(github_base_url=None, temperature_directory=None, cache_directory="data_cache"):
    """Tell the module where the data is.
    github_base_url: raw-file URL prefix of the folder holding the GitHub files, e.g.
        'https://raw.githubusercontent.com/<user>/<repo>/<branch>/<folder>/'
        (each file is downloaded once into cache_directory; delete that folder to force a re-download).
    temperature_directory: folder on this computer containing the two local-temperature CSVs."""
    if github_base_url is not None and not github_base_url.endswith("/"):
        github_base_url += "/"
    _DATA_SOURCES.update(github_base_url=github_base_url, temperature_directory=temperature_directory,
                         cache_directory=cache_directory)

def _path_of_github_file(file_name):
    """Download file_name from the configured GitHub folder (only if not cached yet); return its local path."""
    base_url = _DATA_SOURCES["github_base_url"]
    if base_url is None:
        raise RuntimeError("Call configure_data_sources(github_base_url=...) first.")
    os.makedirs(_DATA_SOURCES["cache_directory"], exist_ok=True)
    cached_path = os.path.join(_DATA_SOURCES["cache_directory"], file_name)
    if not os.path.exists(cached_path):
        urllib.request.urlretrieve(base_url + file_name, cached_path)
    return cached_path

def _path_of_local_temperature_file(file_name):
    """Path of a temperature file on this computer."""
    directory = _DATA_SOURCES["temperature_directory"]
    if directory is None:
        raise RuntimeError("Call configure_data_sources(temperature_directory=...) first.")
    return os.path.join(directory, file_name)


# ======================= grid conversion =======================
def latlon_to_grid_indices(latitudes, longitudes):
    """Latitudes/longitudes of 1-degree pixel centres (longitude in 0..360 OR -180..180)
    -> (row_indices, col_indices) into a (180, 360) grid."""
    longitudes = ((np.asarray(longitudes) + 180) % 360) - 180      # unify the two longitude conventions
    return (np.round(np.asarray(latitudes) + 89.5).astype(int),    # lat -89.5 -> row 0
            np.round(longitudes + 179.5).astype(int))              # lon -179.5 -> column 0

def convert_latlon_table_to_grid(table, value_column, latitude_column="lat", longitude_column="lon",
                                 require_full_coverage=True):
    """Table with lat, lon and a value column -> (180, 360) grid.
    If require_full_coverage is True, raises when any of the 64,800 pixels is absent from the table;
    otherwise absent pixels become NaN."""
    grid = np.full((NUM_LAT_ROWS, NUM_LON_COLS), np.nan, dtype=np.float32)
    rows, cols = latlon_to_grid_indices(table[latitude_column].to_numpy(), table[longitude_column].to_numpy())
    grid[rows, cols] = table[value_column].to_numpy()
    n_missing = int(np.isnan(grid).sum())
    if require_full_coverage and n_missing:
        raise ValueError(f"{n_missing} of {grid.size} pixels are missing from the table")
    return grid


# ======================= loading: map variables and Koppen labels (GitHub) =======================
def load_time_invariant_map_variables(epoch="0ka"):
    """Altitude, ice thickness, land fraction, ice fraction for the chosen epoch ('0ka' or '21ka').
    Returns (180, 360, 4). 'Time invariant' = these maps are treated as constant over the years."""
    table = pd.read_csv(_path_of_github_file("MapVariables.csv"))
    names = [f"altitude_{epoch}_m", f"ice_thickness_{epoch}_m",
             f"land_area_fraction_{epoch}_pct", f"ice_area_fraction_{epoch}_pct"]
    return np.stack([convert_latlon_table_to_grid(table, n) for n in names], axis=-1)

def load_koppen_labels_for_year(year):
    """Returns (koppen_labels, row_indices, col_indices) for every Koppen-labelled pixel in that year.
    Only land pixels have labels (about 25,600 of 64,800)."""
    table = pd.read_csv(_path_of_github_file("koppen_historical_1900_2026.csv"),
                        usecols=["lat", "lon", str(year)])
    rows, cols = latlon_to_grid_indices(table["lat"].to_numpy(), table["lon"].to_numpy())
    return table[str(year)].to_numpy(), rows, cols


# ======================= loading: global temperature / CO2 (GitHub) =======================
def load_global_mean_temperature_and_co2_by_year(scenario="ssp245", last_year=2099):
    """Returns (temperature_by_year, co2_by_year) as dicts {year: value} for 1900..last_year.
    Years up to 2025 come from the historical file; 2026 onward from the chosen SSP scenario
    (ssp119, ssp126, ssp245, ssp370, ssp434, ssp460, ssp585)."""
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

def compute_area_weighted_global_mean(grid):
    """Global mean of a (180, 360) grid, weighting each row by cos(latitude) (pixels shrink toward the poles).
    Use it to build the 'global temperature' feature from your own local grids, so the global and local
    temperatures come from the same source."""
    weights = np.cos(np.deg2rad(np.arange(NUM_LAT_ROWS) - 89.5))[:, None]
    return float((grid * weights).sum() / (weights.sum() * grid.shape[1]))


# ======================= loading: local temperature (your computer) =======================
def _wide_table_to_grids_by_year(table, years):
    """Wide table (one column per year, plus lat/lon) -> {year: (180, 360) grid}."""
    rows, cols = latlon_to_grid_indices(table["lat"].to_numpy(), table["lon"].to_numpy())
    grids_by_year = {}
    for year in years:
        grid = np.full((NUM_LAT_ROWS, NUM_LON_COLS), np.nan, dtype=np.float32)
        grid[rows, cols] = table[str(year)].to_numpy()
        if np.isnan(grid).any():
            raise ValueError(f"Year {year}: temperature table does not cover all pixels")
        grids_by_year[year] = grid
    return grids_by_year

def load_local_temperature_grids_by_year(scenario="ssp245", first_year=1900, last_year=2099):
    """{year: (180, 360) local annual mean temperature in C}.
    1900-2014 come from the historical file, 2015-2099 from the chosen scenario (2015 is the first
    scenario year, so the scenario only matters from 2015)."""
    grids_by_year = {}
    historical_years = [y for y in range(first_year, min(last_year, 2014) + 1)]
    if historical_years:
        historical = pd.read_csv(_path_of_local_temperature_file(LOCAL_HISTORICAL_TEMPERATURE_FILE))
        grids_by_year.update(_wide_table_to_grids_by_year(historical, historical_years))
    scenario_years = [y for y in range(max(first_year, 2015), last_year + 1)]
    if scenario_years:
        # The scenario file is huge (7 scenarios x 64,800 pixels): read it in chunks, keep only our scenario.
        kept_chunks = []
        for chunk in pd.read_csv(_path_of_local_temperature_file(LOCAL_SCENARIO_TEMPERATURE_FILE),
                                 chunksize=NUM_PIXELS, usecols=["scenario", "lat", "lon"] + [str(y) for y in scenario_years]):
            if chunk["scenario"].iloc[0] == scenario:
                kept_chunks.append(chunk)
        if not kept_chunks:
            raise ValueError(f"Scenario '{scenario}' not found in the scenario temperature file")
        grids_by_year.update(_wide_table_to_grids_by_year(pd.concat(kept_chunks), scenario_years))
    return grids_by_year

def compute_period_mean_temperature_grids(local_temperature_grids_by_year, periods=DEFAULT_KOPPEN_PERIODS):
    """Average the yearly grids over each Koppen period.
    Returns [(koppen_label_year, mean_temperature_grid), ...] ready for make_koppen_tree_training_data."""
    return [(label_year, np.mean([local_temperature_grids_by_year[y] for y in range(first, last + 1)], axis=0))
            for label_year, first, last in periods]


# ======================= the kernel =======================
def wrap_grid_across_edges_and_poles(grid, pad_width):
    """(rows, cols, channels) -> (rows + 2*pad, cols + 2*pad, channels).
    Adds pad_width extra rows/columns on every side so kernels at the borders can read their neighbours:
      - left/right edges wrap around (longitude 180 touches longitude -180);
      - past a pole you see the same row you just left, shifted 180 degrees in longitude."""
    half_turn = grid.shape[1] // 2                                               # 180 degrees = half the columns
    rows_beyond_north = np.roll(grid[::-1][:pad_width], half_turn, axis=1)       # top rows, mirrored and shifted
    rows_beyond_south = np.roll(grid[:pad_width][::-1], half_turn, axis=1)       # bottom rows, mirrored and shifted
    with_poles = np.concatenate([rows_beyond_south, grid, rows_beyond_north], axis=0)
    return np.pad(with_poles, ((0, 0), (pad_width, pad_width), (0, 0)), mode="wrap")   # wrap longitude

def build_neighbourhood_features_for_every_pixel(grid, kernel_size):
    """(rows, cols, channels) -> (rows, cols, kernel_size**2 * channels).
    Each pixel gets its OWN kernel: the values of its kernel_size x kernel_size neighbourhood, flattened.
    All pixels are done at once with a sliding window (no Python loop over pixels)."""
    assert kernel_size % 2 == 1, "kernel_size must be odd so the pixel sits in the middle"
    padded = wrap_grid_across_edges_and_poles(grid, kernel_size // 2)
    windows = sliding_window_view(padded, (kernel_size, kernel_size), axis=(0, 1))   # (rows, cols, ch, k, k)
    windows = windows.transpose(0, 1, 3, 4, 2)                                       # (rows, cols, k, k, ch)
    return windows.reshape(grid.shape[0], grid.shape[1], -1)                         # flatten k, k, ch


# ======================= feature builders (all 64,800 pixels, for training or prediction) ==========
def make_temperature_regression_features(kernel_size, global_mean_temperature, co2_ppm, epoch="0ka",
                                         map_variables=None):
    """Features for EVERY pixel (land + ocean), shape (64800, kernel_size**2 * 4 + 2).
    Columns: kernel of [altitude, ice thickness, land fraction, ice fraction], then global temp, CO2.
    Use epoch='21ka' (and 21 ka global temp / CO2) when predicting the past."""
    if map_variables is None:
        map_variables = load_time_invariant_map_variables(epoch)
    kernels = build_neighbourhood_features_for_every_pixel(map_variables, kernel_size).reshape(NUM_PIXELS, -1)
    # global temp and CO2 are the same for every pixel in a given year, and sit OUTSIDE the kernel
    global_columns = np.full((NUM_PIXELS, 2), [global_mean_temperature, co2_ppm], dtype=np.float32)
    return np.hstack([kernels, global_columns])

def make_koppen_tree_features(kernel_size, surface_temperature_grid, epoch="0ka", map_variables=None):
    """Features for EVERY pixel, shape (64800, kernel_size**2 * 5).
    Columns: kernel of the 4 map variables plus surface temperature (all in one 5-channel kernel)."""
    if map_variables is None:
        map_variables = load_time_invariant_map_variables(epoch)
    five_channels = np.concatenate([map_variables, surface_temperature_grid[..., None].astype(np.float32)], -1)
    return build_neighbourhood_features_for_every_pixel(five_channels, kernel_size).reshape(NUM_PIXELS, -1)


# ======================= ready-to-train functions =======================
def make_temperature_regression_training_data(kernel_size, years, global_mean_temperature_by_year,
                                              co2_ppm_by_year, local_temperature_grid_by_year,
                                              map_epoch="0ka", pixels_per_year=None, random_seed=0):
    """Returns (X, y) ready for model.fit(X, y). One sample per pixel per year (land + ocean).
    years: iterable of years. *_by_year: dicts {year: value} (grids are (180, 360)).
    pixels_per_year: if set, keep only this many RANDOM pixels from each year (different pixels every
        year, reproducible via random_seed). All 64,800 pixels x 200 years x 102 features is ~5 GB;
        5,000 per year is ~0.4 GB. Neighbouring pixels are near-duplicates, so little information is lost."""
    rng = np.random.default_rng(random_seed)
    map_variables = load_time_invariant_map_variables(map_epoch)
    # kernels of the time-invariant maps are identical every year: compute once, reuse
    kernels = build_neighbourhood_features_for_every_pixel(map_variables, kernel_size).reshape(NUM_PIXELS, -1)
    X_parts, y_parts = [], []
    for year in years:
        kept_pixels = (np.arange(NUM_PIXELS) if pixels_per_year is None
                       else rng.choice(NUM_PIXELS, pixels_per_year, replace=False))
        global_columns = np.full((len(kept_pixels), 2),
                                 [global_mean_temperature_by_year[year], co2_ppm_by_year[year]], dtype=np.float32)
        X_parts.append(np.hstack([kernels[kept_pixels], global_columns]))
        y_parts.append(local_temperature_grid_by_year[year].reshape(-1)[kept_pixels])   # label: local temperature
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
        X_parts.append(features[rows * NUM_LON_COLS + cols])      # keep only labelled centre pixels
        y_parts.append(labels)
    return np.vstack(X_parts), np.concatenate(y_parts)

def reshape_predictions_to_grid(flat_predictions):
    """(64800,) predictions from the all-pixel feature builders -> (180, 360) grid."""
    return np.asarray(flat_predictions).reshape(NUM_LAT_ROWS, NUM_LON_COLS)


# ======================= one-call versions: load everything and return (X, y) =======================
def make_temperature_regression_training_data_from_files(kernel_size, scenario="ssp245", years=range(1900, 2100),
                                                         pixels_per_year=None, global_temperature_from="file"):
    """Loads all data (GitHub + local temperature files) and returns (X, y) for model.fit.
    global_temperature_from: 'file' = global mean from the co2_temperature files;
                             'grid' = area-weighted mean of the local grids (self-consistent)."""
    years = list(years)
    global_temperature_by_year, co2_by_year = load_global_mean_temperature_and_co2_by_year(scenario, max(years))
    local_grids = load_local_temperature_grids_by_year(scenario, min(years), max(years))
    if global_temperature_from == "grid":
        global_temperature_by_year = {y: compute_area_weighted_global_mean(local_grids[y]) for y in years}
    return make_temperature_regression_training_data(kernel_size, years, global_temperature_by_year, co2_by_year,
                                                     local_grids, pixels_per_year=pixels_per_year)

def make_koppen_tree_training_data_from_files(kernel_size, scenario="ssp245", periods=DEFAULT_KOPPEN_PERIODS):
    """Loads all data and returns (X, y) for model.fit. Temperature channel = mean over each Koppen period."""
    grids = load_local_temperature_grids_by_year(scenario, min(p[1] for p in periods), max(p[2] for p in periods))
    return make_koppen_tree_training_data(kernel_size, compute_period_mean_temperature_grids(grids, periods))
