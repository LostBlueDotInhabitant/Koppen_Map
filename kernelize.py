"""Neighbourhood (kernel) features for the Koppen decision tree and the local-temperature regression.

DATA LAYOUT
  * On GitHub (downloaded once, then cached on disk) OR just placed in the same folder as the two temperature
    files below: MapVariables.csv, koppen_historical_1900_2026.csv,
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

ALL_SCENARIOS = ("ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585")

# Koppen labels only change every 30 years, so there are 4 distinct label maps in 1900-2026.
# Each entry: (year whose Koppen label to use, first year, last year of the period to average temperature over)
DEFAULT_KOPPEN_PERIODS = [(1900, 1900, 1930), (1931, 1931, 1960), (1961, 1961, 1990), (1991, 1991, 2020)]

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
    """Set the defaults used by every function below (each call sets ALL of them).
    kernel_size: odd integer, width/height of the neighbourhood (3, 5, 7, ...).
    water_vector_kernel_size: neighbourhood used for the water AND ice vectors; None = same as kernel_size.
        (A larger window lets the vectors 'see' water/ice further away than the kernel does.)
    include_water_vectors: add the 12 water/ice vector columns to the features of both models.
    water_inverse_distance_power: power of the distance-weighted vector in the water AND ice vector sets
        (default 2 = 1/distance**2). The other vector of each set is not weighted by distance (power 0).
    inverse_distance_power: fallback power used only when compute_vectors_toward_marked_pixels is called
        directly without a power. Not used by the water or ice vectors.
    scale_east_west_by_cos_latitude: False -> distances in grid-pixel units;
        True -> east-west offsets shrunk by cos(latitude), i.e. closer to true distances near the poles."""
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
    cached_path = os.path.join(_DATA_SOURCES["cache_directory"], file_name)
    if os.path.exists(cached_path):
        return cached_path
    # also accept the file sitting in the same folder as the temperature CSVs (no GitHub needed)
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
    """{year: (180, 360) local annual mean temperature in C} for ONE scenario.
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

def load_all_local_temperature_grids(scenarios=ALL_SCENARIOS, first_year=1900, last_year=2099):
    """Returns (historical_grids {year: grid} for years <= 2014, scenario_grids {scenario: {year: grid}} for
    years >= 2015). The big scenario file is read ONCE for all the requested scenarios."""
    historical_grids = {}
    if first_year <= 2014:
        historical_years = range(first_year, min(last_year, 2014) + 1)
        historical = pd.read_csv(_path_of_local_temperature_file(LOCAL_HISTORICAL_TEMPERATURE_FILE))
        historical_grids = _wide_table_to_grids_by_year(historical, historical_years)
    scenario_grids = {}
    scenario_years = list(range(max(first_year, 2015), last_year + 1))
    if scenario_years:
        for chunk in pd.read_csv(_path_of_local_temperature_file(LOCAL_SCENARIO_TEMPERATURE_FILE),
                                 chunksize=NUM_PIXELS,
                                 usecols=["scenario", "lat", "lon"] + [str(y) for y in scenario_years]):
            name = chunk["scenario"].iloc[0]
            if name in scenarios:
                scenario_grids[name] = _wide_table_to_grids_by_year(chunk, scenario_years)
        missing = set(scenarios) - set(scenario_grids)
        if missing:
            raise ValueError(f"Scenarios not found in the scenario temperature file: {sorted(missing)}")
    return historical_grids, scenario_grids

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


# ======================= water and ice vectors =======================
def compute_vectors_toward_marked_pixels(marked_pixel_grid, kernel_size=None, inverse_distance_power=None,
                                         scale_east_west_by_cos_latitude=None):
    """For every pixel, summarise WHERE the marked pixels (marked_pixel_grid > 0, e.g. water or ice) are
    around it with two vectors. Returns (180, 360, 4):
        [0, 1] unit-vector sum   (east, north): sum over marked pixels in the window of d / |d|
               (no distance weighting, i.e. power 0)
        [2, 3] inverse-distance  (east, north): sum over marked pixels of (d / |d|) / |d|**power
    d = displacement from the central pixel to the marked pixel (east = +, north = +); the central pixel
    itself is excluded. The window wraps like the kernel does.
    A (0, 0) vector means either no marked pixel in the window or perfectly balanced ones; the kernel columns
    tell the model which."""
    k = _resolve_kernel_size(_KERNEL_SETTINGS["water_vector_kernel_size"]) if kernel_size is None else kernel_size
    power = _KERNEL_SETTINGS["inverse_distance_power"] if inverse_distance_power is None else inverse_distance_power
    scale_ew = (_KERNEL_SETTINGS["scale_east_west_by_cos_latitude"] if scale_east_west_by_cos_latitude is None
                else scale_east_west_by_cos_latitude)
    assert k % 2 == 1, "vector kernel size must be odd"
    radius = k // 2
    is_marked = (marked_pixel_grid > 0).astype(np.float32)                        # (180, 360), 1 = marked
    padded = wrap_grid_across_edges_and_poles(is_marked[..., None], radius)[..., 0]
    windows = sliding_window_view(padded, (k, k))                                  # (180, 360, k, k)
    offsets = np.arange(-radius, radius + 1)
    north_offset, east_offset = np.meshgrid(offsets, offsets, indexing="ij")       # window row i -> north offset
    # displacement of every window cell from the centre, one (k, k) table per latitude row
    cos_latitude = np.cos(np.deg2rad(np.arange(NUM_LAT_ROWS) - 89.5)) if scale_ew else np.ones(NUM_LAT_ROWS)
    east = east_offset[None, :, :] * cos_latitude[:, None, None]                   # (180, k, k)
    north = np.broadcast_to(north_offset[None, :, :], east.shape).astype(float)
    distance = np.sqrt(east ** 2 + north ** 2)
    is_centre = distance == 0
    safe_distance = np.where(is_centre, 1.0, distance)                              # avoid dividing by 0 at the centre
    unit_east = np.where(is_centre, 0.0, east / safe_distance)                      # d / |d|
    unit_north = np.where(is_centre, 0.0, north / safe_distance)
    weight = safe_distance ** -power                                                # 1 / |d|**power
    # sum the weights over the window cells that are marked (einsum = weighted sum per pixel)
    def weighted_sum(per_row_weights):
        return np.einsum("rcij,rij->rc", windows, per_row_weights.astype(np.float32))
    return np.stack([weighted_sum(unit_east), weighted_sum(unit_north),
                     weighted_sum(unit_east * weight), weighted_sum(unit_north * weight)], axis=-1)


def compute_water_vectors_for_every_pixel(land_fraction_grid, **options):
    """Water vectors toward ALL water pixels (land fraction == 0), including water under ice (ice shelves).
    Returns (180, 360, 4): unit-sum (east, north) [no distance weighting],
    inverse-distance-sum (east, north) [1/distance**water_inverse_distance_power, default 2]."""
    options.setdefault("inverse_distance_power", _KERNEL_SETTINGS["water_inverse_distance_power"])
    return compute_vectors_toward_marked_pixels(land_fraction_grid == 0, **options)

def compute_open_water_vectors_for_every_pixel(land_fraction_grid, ice_fraction_grid, **options):
    """Same, but only toward ICE-FREE water: land fraction == 0 AND ice fraction == 0. Water under ice
    (ice shelves) is ignored, as if it were neither land nor water. Returns (180, 360, 4)."""
    options.setdefault("inverse_distance_power", _KERNEL_SETTINGS["water_inverse_distance_power"])
    return compute_vectors_toward_marked_pixels((land_fraction_grid == 0) & (ice_fraction_grid == 0), **options)

def compute_ice_vectors_for_every_pixel(ice_fraction_grid, **options):
    """Ice vectors toward ice pixels (ice fraction > 0, i.e. land ice and ice shelves), built exactly like the
    water vectors: unit-sum (east, north) with no distance weighting, and inverse-distance-sum (east, north)
    weighted by 1/distance**water_inverse_distance_power (default 2). Returns (180, 360, 4)."""
    options.setdefault("inverse_distance_power", _KERNEL_SETTINGS["water_inverse_distance_power"])
    return compute_vectors_toward_marked_pixels(ice_fraction_grid > 0, **options)


# ======================= feature builders (all 64,800 pixels, for training or prediction) ==========
LOCATION_FEATURE_NAMES = ("latitude", "abs_latitude", "sin_lat", "cos_lat", "sin_lat^2", "sin_lat^3",
                          "sin_lon", "cos_lon")

def compute_location_features_for_every_pixel():
    """Where the pixel is on the globe: (64800, 8) float32 columns, in LOCATION_FEATURE_NAMES order.
    The kernel only sees altitude / ice / land fraction, so WITHOUT these the model cannot know a pixel's
    latitude, which is the strongest single driver of mean temperature (held-out R2 ~0.73 -> ~0.97).
    Longitude is given as sin/cos so that -180 and +180 are neighbours."""
    latitude = np.repeat(np.arange(NUM_LAT_ROWS) - 89.5, NUM_LON_COLS)
    longitude = np.tile(np.arange(NUM_LON_COLS) - 179.5, NUM_LAT_ROWS)
    sin_lat = np.sin(np.deg2rad(latitude))
    return np.stack([latitude, np.abs(latitude), sin_lat, np.cos(np.deg2rad(latitude)), sin_lat ** 2, sin_lat ** 3,
                     np.sin(np.deg2rad(longitude)), np.cos(np.deg2rad(longitude))], axis=1).astype(np.float32)

def _map_features_for_every_pixel(map_variables, kernel_size, extra_channel=None, include_location_features=False):
    """Time-invariant part of the features: kernel columns, then (if enabled) 12 vector columns:
    4 toward all water, 4 toward ice-free water, 4 toward ice (each set = unweighted pair + 1/d**2 pair),
    then (if include_location_features) 8 latitude/longitude columns (the temperature model uses these).
    extra_channel: optional (180, 360) grid (surface temperature) appended as a 5th kernel channel."""
    kernel_size = _resolve_kernel_size(kernel_size)
    channels = map_variables if extra_channel is None else np.concatenate(
        [map_variables, extra_channel[..., None].astype(np.float32)], axis=-1)
    columns = [build_neighbourhood_features_for_every_pixel(channels, kernel_size).reshape(NUM_PIXELS, -1)]
    if _KERNEL_SETTINGS["include_water_vectors"]:
        land_fraction, ice_fraction = map_variables[..., 2], map_variables[..., 3]   # channels 2 and 3
        columns.append(compute_water_vectors_for_every_pixel(land_fraction).reshape(NUM_PIXELS, -1))
        columns.append(compute_open_water_vectors_for_every_pixel(land_fraction, ice_fraction)
                       .reshape(NUM_PIXELS, -1))
        columns.append(compute_ice_vectors_for_every_pixel(ice_fraction).reshape(NUM_PIXELS, -1))
    if include_location_features:
        columns.append(compute_location_features_for_every_pixel())
    return np.hstack(columns).astype(np.float32, copy=False)

def make_temperature_regression_features(kernel_size, global_mean_temperature, co2_ppm, epoch="0ka",
                                         map_variables=None):
    """Features for EVERY pixel (land + ocean). kernel_size=None -> configured default.
    Columns: kernel of [altitude, ice thickness, land fraction, ice fraction] (k*k*4),
    the 12 vector columns (if enabled: 4 all-water, 4 ice-free-water, 4 ice), the 8 location columns
    (latitude etc.), then global temp, CO2.
    Use epoch='21ka' (and 21 ka global temp / CO2) when predicting the past."""
    if map_variables is None:
        map_variables = load_time_invariant_map_variables(epoch)
    kernels = _map_features_for_every_pixel(map_variables, kernel_size, include_location_features=True)
    # global temp and CO2 are the same for every pixel in a given year, and sit OUTSIDE the kernel
    global_columns = np.full((NUM_PIXELS, 2), [global_mean_temperature, co2_ppm], dtype=np.float32)
    return np.hstack([kernels, global_columns])

def make_koppen_tree_features(kernel_size, surface_temperature_grid, epoch="0ka", map_variables=None):
    """Features for EVERY pixel. kernel_size=None -> configured default.
    Columns: kernel of the 4 map variables plus surface temperature (one 5-channel kernel, k*k*5),
    then the 12 vector columns (if enabled: 4 all-water, 4 ice-free-water, 4 ice)."""
    if map_variables is None:
        map_variables = load_time_invariant_map_variables(epoch)
    return _map_features_for_every_pixel(map_variables, kernel_size, extra_channel=surface_temperature_grid)


# ======================= ready-to-train functions =======================
def make_temperature_regression_training_data(kernel_size, years, global_mean_temperature_by_year,
                                              co2_ppm_by_year, local_temperature_grid_by_year,
                                              map_epoch="0ka", pixels_per_year=None, random_seed=0,
                                              precomputed_kernels=None):
    """Returns (X, y) ready for model.fit(X, y). One sample per pixel per year (land + ocean).
    kernel_size: odd int, or None to use the value set with configure_kernel().
    years: iterable of years. *_by_year: dicts {year: value} (grids are (180, 360)).
    pixels_per_year: if set, keep only this many RANDOM pixels from each year (different pixels every
        year, reproducible via random_seed). All 64,800 pixels x 200 years x 102 features is ~5 GB;
        5,000 per year is ~0.4 GB. Neighbouring pixels are near-duplicates, so little information is lost.
    precomputed_kernels: optional output of _map_features_for_every_pixel(..., include_location_features=True),
        to avoid recomputing it."""
    rng = np.random.default_rng(random_seed)
    if precomputed_kernels is None:
        map_variables = load_time_invariant_map_variables(map_epoch)
        # kernels (and vectors) of the time-invariant maps are identical every year: compute once, reuse
        precomputed_kernels = _map_features_for_every_pixel(map_variables, kernel_size,
                                                            include_location_features=True)
    kernels = precomputed_kernels
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


# ======================= one-call versions: load everything and return the data =======================
def make_temperature_regression_training_data_from_files(kernel_size=None, scenarios=ALL_SCENARIOS,
                                                         years=range(1900, 2100), pixels_per_year=None,
                                                         global_temperature_from="file", random_seed=0):
    """Loads all data (GitHub + local temperature files) and returns (X, y, year_of_row, scenario_of_row)
    covering ALL the scenarios stacked together.
    Historical years (<= 2014) are included once (scenario_of_row = 'historical'); every scenario contributes
    its own 2015+ years. Then split into train/test by picking random rows (random_train_test_split).
    scenarios: default = all 7 SSP scenarios.
    pixels_per_year: random pixels kept per year (a different random sample for every scenario and year).
    global_temperature_from: 'file' = global mean from the co2_temperature files;
                             'grid' = area-weighted mean of the local grids (self-consistent)."""
    years, scenarios = list(years), list(scenarios)
    historical_years = [y for y in years if y <= 2014]
    future_years = [y for y in years if y >= 2015]
    historical_grids, scenario_grids = load_all_local_temperature_grids(scenarios, min(years), max(years))
    # the time-invariant kernels and vectors are the same for every scenario: compute once
    kernels = _map_features_for_every_pixel(load_time_invariant_map_variables("0ka"), kernel_size,
                                            include_location_features=True)
    rows_per_year = NUM_PIXELS if pixels_per_year is None else pixels_per_year
    X_parts, y_parts, year_parts, scenario_parts = [], [], [], []
    for i, scenario in enumerate(scenarios):
        run_years = (historical_years if i == 0 else []) + future_years     # historical years only once
        if not run_years:
            continue
        grids = {**historical_grids, **scenario_grids.get(scenario, {})}
        temperature_by_year, co2_by_year = load_global_mean_temperature_and_co2_by_year(scenario, max(years))
        if global_temperature_from == "grid":
            temperature_by_year = {y: compute_area_weighted_global_mean(grids[y]) for y in run_years}
        X, y = make_temperature_regression_training_data(
            kernel_size, run_years, temperature_by_year, co2_by_year, grids,
            pixels_per_year=pixels_per_year, random_seed=random_seed + i, precomputed_kernels=kernels)
        year_of_row = np.repeat(run_years, rows_per_year)
        X_parts.append(X)
        y_parts.append(y)
        year_parts.append(year_of_row)
        scenario_parts.append(np.where(year_of_row <= 2014, "historical", scenario))
    return (np.vstack(X_parts), np.concatenate(y_parts),
            np.concatenate(year_parts), np.concatenate(scenario_parts))

def make_koppen_tree_training_data_from_files(kernel_size=None, scenario="ssp245", periods=DEFAULT_KOPPEN_PERIODS):
    """Loads all data and returns (X, y) for model.fit. Temperature channel = mean over each Koppen period."""
    grids = load_local_temperature_grids_by_year(scenario, min(p[1] for p in periods), max(p[2] for p in periods))
    return make_koppen_tree_training_data(kernel_size, compute_period_mean_temperature_grids(grids, periods))

def random_train_test_split(X, y, test_fraction=0.2, random_seed=0, *extra_row_arrays):
    """Randomly splits rows into train and test. Returns X_train, y_train, X_test, y_test, then
    (train, test) pairs for each extra per-row array passed in (e.g. year_of_row, scenario_of_row)."""
    order = np.random.default_rng(random_seed).permutation(len(y))
    n_test = int(len(y) * test_fraction)
    test, train = order[:n_test], order[n_test:]
    result = [X[train], y[train], X[test], y[test]]
    for extra in extra_row_arrays:
        result += [extra[train], extra[test]]
    return tuple(result)
