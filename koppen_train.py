#!/usr/bin/env python3
"""
Köppen pipeline: kernelize-style spatial/vector/latitude features + LightGBM multiclass training.

What changed in this version (aimed at accuracy)
------------------------------------------------
* Early stopping now validates on a spread of held-out EVEN years from every period and scenario (--val-every).
  Before, it validated only on the last scenario years, which could stop training after a few dozen rounds and leave
  an under-fitted model.
* New features, APPENDED after the old columns (use --no-extra-features for the old 258-column layout):
  8 ray distances to water, 8 upwind barrier heights, 15x15 / 31x31 land + altitude means, relief (21 static), and
  trailing 10y / 30y mean temperature, their difference, and local-minus-zonal temperature (4 per year).
  These stand in for precipitation and continentality, which the model cannot otherwise see. No longitude.
* Bigger model: 127 leaves, up to 600 rounds, 3M rows, min_data_in_leaf 40, L2 1.0.
* --device gpu by default (falls back to CPU automatically if LightGBM has no GPU support).
* Single-pass training with early stopping, progress bar, plain file names, and sanity checks are unchanged.
* The saved .joblib records extra_features / ray_max_steps. koppen_predict.py must build the same extra columns
  (build_extra_static_features and the per-year temperature extras in koppen_train.py), or use --no-extra-features.

Constraints enforced:
  * Pixel latitude (lat) is included as a feature (when --no-lat is not passed).
  * Longitude (lon) is NEVER included as a model feature or spatial hold-out axis.

Example (run from the folder holding the scripts and a "data" folder):
    python koppen_train.py
"""

import argparse
import json
import os
import time
import urllib.request

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

N_LAT, N_LON = 180, 360
N_PIXELS = N_LAT * N_LON
DEFAULT_KERNEL_SIZE = 7
ALL_SCENARIOS = ("ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585")
DEFAULT_KOPPEN_PERIODS = [
    (1900, 1900, 1930),
    (1931, 1931, 1960),
    (1961, 1961, 1990),
    (1991, 1991, 2020),
]
SEED = 42

# --------------------------------------------------------------------------------------
# Input files (looked up in --data / --temperature-dir, or downloaded from --github-base-url)
# --------------------------------------------------------------------------------------
MAP_VARIABLES_FILE = "MapVariables.csv"
HISTORICAL_TEMPERATURE_FILE = "tas_annual_mean_1deg_wide_celsius.csv"
SCENARIO_TEMPERATURE_FILE = "temp_annual_all_scenarios_2015_2099.csv"
HISTORICAL_KOPPEN_FILE = "koppen_historical_1900_2026.csv"
SCENARIO_KOPPEN_FILE = "koppen_scenarios_2026_2100.csv"


try:
    from tqdm.auto import tqdm
except ImportError:  # minimal fallback if tqdm is not installed
    tqdm = None


def log(*args):
    msg = " ".join(str(a) for a in (time.strftime("%H:%M:%S"), *args))
    if tqdm is not None:
        tqdm.write(msg)  # prints above any active progress bar instead of breaking it
    else:
        print(msg, flush=True)


class TrainProgress:
    """LightGBM callback: one progress bar over all boosting rounds."""

    order = 30  # run after LightGBM's own callbacks
    before_iteration = False

    def __init__(self, total_rounds, desc="training"):
        self.total = max(1, int(total_rounds))
        self.done = 0
        self.t0 = time.time()
        self.desc = desc
        self.last_print = 0.0
        self.bar = (
            tqdm(total=self.total, desc=desc, unit="round", dynamic_ncols=True,
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")
            if tqdm is not None else None
        )

    def __call__(self, env):
        self.done += 1
        metrics = {}
        for _, name, value, _ in (env.evaluation_result_list or []):
            metrics[name] = f"{value:.4f}"
        if self.bar is not None:
            if metrics:
                self.bar.set_postfix(metrics, refresh=False)
            self.bar.update(1)
        else:  # fallback: single-line text bar, refreshed at most 4x/second
            now = time.time()
            if now - self.last_print > 0.25 or self.done >= self.total:
                self.last_print = now
                frac = min(1.0, self.done / self.total)
                eta = (now - self.t0) / max(self.done, 1) * (self.total - self.done)
                bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
                print(f"\r{self.desc}: {frac:6.1%} [{bar}] {self.done}/{self.total} ETA {eta:5.0f}s", end="", flush=True)

    def close(self):
        if self.bar is not None:
            self.bar.close()
        else:
            print(flush=True)


def year_cols(columns):
    return {int(c): c for c in columns if str(c).isdigit()}


def latlon_to_grid_indices(latitudes, longitudes):
    """Map 1-degree pixel centres (lon in 0..360 or -180..180) -> (row_indices, col_indices) in (180, 360)."""
    lon_norm = ((np.asarray(longitudes, dtype=np.float64) + 180.0) % 360.0) - 180.0
    r = np.round(np.asarray(latitudes, dtype=np.float64) + 89.5).astype(int)
    c = np.round(lon_norm + 179.5).astype(int)
    assert r.min() >= 0 and r.max() < N_LAT and c.min() >= 0 and c.max() < N_LON, "unexpected lat/lon grid"
    return r, c


def to_pix(lat, lon):
    r, c = latlon_to_grid_indices(lat, lon)
    return r * N_LON + c


def resolve_input_file(file_name, local_dirs, github_base_url=None, cache_dir="data_cache"):
    """Find file_name in local_dirs first; optionally download from github_base_url into cache_dir."""
    for d in local_dirs:
        if d:
            p = os.path.join(d, file_name)
            if os.path.exists(p):
                return p
    if github_base_url:
        base = github_base_url if github_base_url.endswith("/") else github_base_url + "/"
        os.makedirs(cache_dir, exist_ok=True)
        cached = os.path.join(cache_dir, file_name)
        if not os.path.exists(cached):
            log(f"downloading {file_name} from {base}")
            urllib.request.urlretrieve(base + file_name, cached)
        return cached
    raise FileNotFoundError(f"Could not find '{file_name}' in {local_dirs} and no --github-base-url provided.")


def load_map_variables_grid(path, epoch="0ka"):
    """Load altitude, ice_thickness, land_area_fraction, ice_area_fraction for epoch ('0ka' or '21ka') -> (180, 360, 4)."""
    table = pd.read_csv(path)
    cols = [
        f"altitude_{epoch}_m",
        f"ice_thickness_{epoch}_m",
        f"land_area_fraction_{epoch}_pct",
        f"ice_area_fraction_{epoch}_pct",
    ]
    r, c = latlon_to_grid_indices(table["lat"].to_numpy(), table["lon"].to_numpy())
    grid = np.full((N_LAT, N_LON, 4), np.nan, dtype=np.float32)
    for ch_i, col in enumerate(cols):
        grid[r, c, ch_i] = table[col].to_numpy(np.float32)
    assert not np.isnan(grid).any(), f"missing pixels in MapVariables.csv for epoch={epoch}"
    return grid


def wrap_grid_across_edges_and_poles(grid, pad_width):
    """(rows, cols, channels) -> (rows + 2*pad, cols + 2*pad, channels) with pole & dateline wrap."""
    half_turn = grid.shape[1] // 2
    rows_beyond_north = np.roll(grid[::-1][:pad_width], half_turn, axis=1)
    rows_beyond_south = np.roll(grid[:pad_width][::-1], half_turn, axis=1)
    with_poles = np.concatenate([rows_beyond_south, grid, rows_beyond_north], axis=0)
    return np.pad(with_poles, ((0, 0), (pad_width, pad_width), (0, 0)), mode="wrap")


def build_neighbourhood_features_for_every_pixel(grid, kernel_size=DEFAULT_KERNEL_SIZE):
    """(rows, cols, channels) -> (rows, cols, kernel_size**2 * channels)."""
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    padded = wrap_grid_across_edges_and_poles(grid, kernel_size // 2)
    windows = sliding_window_view(padded, (kernel_size, kernel_size), axis=(0, 1))
    windows = windows.transpose(0, 1, 3, 4, 2)
    return windows.reshape(grid.shape[0], grid.shape[1], -1)


def compute_vectors_toward_marked_pixels(
    marked_pixel_grid,
    kernel_size=DEFAULT_KERNEL_SIZE,
    inverse_distance_power=2,
    scale_east_west_by_cos_latitude=False,
):
    """Returns (180, 360, 4): unweighted (east, north) sum + inverse-distance (east, north) sum."""
    assert kernel_size % 2 == 1, "vector kernel_size must be odd"
    radius = kernel_size // 2
    is_marked = (marked_pixel_grid > 0).astype(np.float32)
    padded = wrap_grid_across_edges_and_poles(is_marked[..., None], radius)[..., 0]
    windows = sliding_window_view(padded, (kernel_size, kernel_size))

    offsets = np.arange(-radius, radius + 1)
    north_offset, east_offset = np.meshgrid(offsets, offsets, indexing="ij")
    cos_lat = (
        np.cos(np.deg2rad(np.arange(N_LAT) - 89.5))
        if scale_east_west_by_cos_latitude
        else np.ones(N_LAT)
    )
    east = east_offset[None, :, :] * cos_lat[:, None, None]
    north = np.broadcast_to(north_offset[None, :, :], east.shape).astype(np.float64)
    distance = np.sqrt(east**2 + north**2)
    is_centre = distance == 0
    safe_dist = np.where(is_centre, 1.0, distance)
    unit_east = np.where(is_centre, 0.0, east / safe_dist)
    unit_north = np.where(is_centre, 0.0, north / safe_dist)
    weight = safe_dist ** (-inverse_distance_power)

    def weighted_sum(w):
        return np.einsum("rcij,rij->rc", windows, w.astype(np.float32))

    return np.stack(
        [
            weighted_sum(unit_east),
            weighted_sum(unit_north),
            weighted_sum(unit_east * weight),
            weighted_sum(unit_north * weight),
        ],
        axis=-1,
    )


def build_static_spatial_features(
    map_grid_4ch,
    kernel_size=DEFAULT_KERNEL_SIZE,
    vector_kernel_size=DEFAULT_KERNEL_SIZE,
    include_water_vectors=True,
    water_power=2,
    scale_ew=False,
    include_latitude=True,
):
    """Build per-pixel static features: map kernel + optional 12 water/ice vectors + optional latitude (no longitude)."""
    parts = [build_neighbourhood_features_for_every_pixel(map_grid_4ch, kernel_size).reshape(N_PIXELS, -1)]
    if include_water_vectors:
        land_frac = map_grid_4ch[..., 2]
        ice_frac = map_grid_4ch[..., 3]
        all_water = compute_vectors_toward_marked_pixels(
            land_frac == 0, vector_kernel_size, water_power, scale_ew
        ).reshape(N_PIXELS, -1)
        open_water = compute_vectors_toward_marked_pixels(
            (land_frac == 0) & (ice_frac == 0), vector_kernel_size, water_power, scale_ew
        ).reshape(N_PIXELS, -1)
        ice_vec = compute_vectors_toward_marked_pixels(
            ice_frac > 0, vector_kernel_size, water_power, scale_ew
        ).reshape(N_PIXELS, -1)
        parts.extend([all_water, open_water, ice_vec])
    if include_latitude:
        lat_of_row = np.arange(N_LAT, dtype=np.float32) - 89.5
        parts.append(np.repeat(lat_of_row, N_LON).reshape(N_PIXELS, 1))
    return np.hstack(parts).astype(np.float32)


N_EXTRA_STATIC = 21   # 8 ray distances to water + 8 upwind barrier heights + 4 multi-scale means + 1 relief
N_EXTRA_TEMP = 4      # trailing 10y mean, trailing 30y mean, 10y-30y trend, local minus zonal-mean temperature
RAY_DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]  # (north, east) steps


def box_mean(channel, size):
    """Mean over a size x size window around every pixel (pole + dateline wrap). channel: (rows, cols)."""
    padded = wrap_grid_across_edges_and_poles(channel[..., None].astype(np.float32), size // 2)[..., 0]
    return sliding_window_view(padded, (size, size)).mean(axis=(-1, -2)).astype(np.float32)


def build_extra_static_features(map_grid_4ch, max_steps=30, alt_steps=12):
    """Large-scale context that a 7x7 kernel cannot see (a proxy for precipitation, which is not an input):
    * distance (in pixels, capped) to the nearest water along 8 compass rays  -> coast distance / continentality
    * height of the highest terrain within `alt_steps` pixels along 8 rays, relative to the pixel -> rain shadows
    * 15x15 and 31x31 mean land fraction and altitude, and local relief
    Longitude is NOT used. Returns (N_PIXELS, N_EXTRA_STATIC)."""
    alt, land = map_grid_4ch[..., 0], map_grid_4ch[..., 2]
    pad = max(max_steps, alt_steps)
    stacked = np.stack([(land == 0).astype(np.float32), alt], axis=-1)
    padded = wrap_grid_across_edges_and_poles(stacked, pad)

    def view(dn, de, s, ch):
        return padded[pad + dn * s: pad + dn * s + N_LAT, pad + de * s: pad + de * s + N_LON, ch]

    dists, barriers = [], []
    for dn, de in RAY_DIRECTIONS:
        dist = np.full((N_LAT, N_LON), max_steps + 1, dtype=np.float32)
        for s in range(max_steps, 0, -1):          # far -> near, so the nearest water wins
            dist = np.where(view(dn, de, s, 0) > 0, np.float32(s), dist)
        dists.append(dist)
        top = np.full((N_LAT, N_LON), -np.inf, dtype=np.float32)
        for s in range(1, alt_steps + 1):
            top = np.maximum(top, view(dn, de, s, 1))
        barriers.append(top - alt)
    land15, land31 = box_mean(land, 15), box_mean(land, 31)
    alt15, alt31 = box_mean(alt, 15), box_mean(alt, 31)
    cols = dists + barriers + [land15, land31, alt15, alt31, alt - alt15]
    out = np.stack(cols, axis=-1).reshape(N_PIXELS, -1).astype(np.float32)
    assert out.shape[1] == N_EXTRA_STATIC
    return out


def trailing_mean(arr, window):
    """arr: (pixels, years) with NaN gaps -> mean of the last `window` years up to and including each year."""
    valid = ~np.isnan(arr)
    cs = np.concatenate([np.zeros((arr.shape[0], 1)), np.cumsum(np.where(valid, arr, 0.0), axis=1, dtype=np.float64)], axis=1)
    cn = np.concatenate([np.zeros((arr.shape[0], 1)), np.cumsum(valid, axis=1, dtype=np.float64)], axis=1)
    hi = np.arange(arr.shape[1]) + 1
    lo = np.maximum(hi - window, 0)
    n = cn[:, hi] - cn[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[:, hi] - cs[:, lo]) / n
    out[n == 0] = np.nan
    return out.astype(np.float32)


def read_wide_temperature(path, scenario_col=False):
    header = pd.read_csv(path, nrows=0).columns
    ycols = year_cols(header)
    names = [ycols[y] for y in sorted(ycols)]
    use = (["scenario"] if scenario_col else []) + ["lat", "lon"] + names
    out = {}
    for ch in pd.read_csv(path, usecols=use, dtype={c: np.float32 for c in names}, chunksize=200_000):
        groups = ch.groupby("scenario") if scenario_col else [("-", ch)]
        for s, g in groups:
            if s not in out:
                out[s] = np.full((N_PIXELS, len(names)), np.nan, dtype=np.float32)
            out[s][to_pix(g["lat"], g["lon"])] = g[names].to_numpy()
    return out, {y: i for i, y in enumerate(sorted(ycols))}


def read_koppen(path, scenario_col=False):
    header = pd.read_csv(path, nrows=0).columns
    ycols = year_cols(header)
    names = [ycols[y] for y in sorted(ycols)]
    use = (["scenario"] if scenario_col else []) + ["lat", "lon"] + names
    parts = {}
    for ch in pd.read_csv(path, usecols=use, chunksize=100_000):
        groups = ch.groupby("scenario") if scenario_col else [("-", ch)]
        for s, g in groups:
            parts.setdefault(s, []).append(
                (to_pix(g["lat"], g["lon"]), g[names].fillna("").to_numpy().astype("<U4"))
            )
    out = {
        s: (np.concatenate([p for p, _ in v]), np.vstack([l for _, l in v]))
        for s, v in parts.items()
    }
    return out, {y: i for i, y in enumerate(sorted(ycols))}


def predict_codes(booster, X, chunk=400_000):
    out = np.empty(len(X), dtype=np.int16)
    for i in range(0, len(X), chunk):
        out[i : i + chunk] = booster.predict(X[i : i + chunk]).argmax(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="Directory containing local CSVs")
    ap.add_argument("--temperature-dir", default=None, help="Optional separate directory for temperature CSVs")
    ap.add_argument("--github-base-url", default=None, help="Optional raw GitHub base URL fallback")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--out", default="models", help="Output directory for model and metrics")
    ap.add_argument("--epoch", choices=["0ka", "21ka"], default="0ka")
    ap.add_argument("--kernel-size", type=int, default=DEFAULT_KERNEL_SIZE)
    ap.add_argument("--vector-kernel-size", type=int, default=None)
    ap.add_argument("--no-water-vectors", action="store_true")
    ap.add_argument("--no-lat", action="store_true", help="Disable latitude feature (longitude is never used)")
    ap.add_argument("--water-power", type=int, default=2)
    ap.add_argument("--scale-ew-cos-lat", action="store_true")
    ap.add_argument("--train-end", type=int, default=2050)
    ap.add_argument("--test-end", type=int, default=2099)
    ap.add_argument("--bridge-scenario", default="ssp245")
    ap.add_argument("--train-scenarios", default="all")
    ap.add_argument("--n-estimators", type=int, default=600, help="max boosting rounds (early stopping may use fewer)")
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-data-in-leaf", type=int, default=40)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--max-train-rows", type=int, default=3_000_000, help="0 = use every selected row")
    ap.add_argument("--year-stride", type=int, default=1)
    ap.add_argument("--even-years", action=argparse.BooleanOptionalAction, default=True,
                    help="train on the EVEN years 1900-2099 (historical, bridge and every scenario) and test on the "
                         "held-out ODD years. Implies --no-anchor: with an anchor from the previous (even) year, "
                         "every odd test year would just be a copy of the year before. ON by default; use --no-even-years "
                         "for the old anchor model (train up to --train-end, test after).")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--max-lag", type=int, default=49)
    ap.add_argument("--change-weight", type=float, default=1.0,
                    help="weight of rows whose label differs from the anchor (keep 1.0: larger values cause false changes)")
    ap.add_argument("--holdout-frac", type=float, default=0.0, help="Fraction of latitude bands held out")
    ap.add_argument("--holdout-block-deg", type=int, default=30)
    ap.add_argument("--holdout-scenarios", default="")
    ap.add_argument("--val-years", type=int, default=5, help="last N scenario years before --train-end used for early stopping")
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=5,
                    help="even-years mode: every Nth even year (all periods, all scenarios) is held out for early stopping")
    ap.add_argument("--no-extra-features", action="store_true",
                    help="drop the large-scale context + temperature-history features (old 258-column layout)")
    ap.add_argument("--ray-max-steps", type=int, default=30, help="max distance (pixels) searched for water along each ray")
    ap.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="gpu",
                    help="LightGBM device; falls back to cpu automatically if the GPU build/driver is unavailable")
    args = ap.parse_args()

    even_mode = args.even_years
    if even_mode:
        args.no_anchor = True
        args.train_end = 2099
        args.test_end = 2099
        log("even-years mode: training on even years 1900-2099, testing on held-out odd years (no anchor feature)")
    assert args.kernel_size % 2 == 1, "kernel_size must be odd"
    vec_k = args.kernel_size if args.vector_kernel_size is None else args.vector_kernel_size
    assert vec_k % 2 == 1, "vector_kernel_size must be odd"
    assert args.train_end >= 2025, "train_end must be >= 2025"
    assert args.year_stride >= 1
    assert args.val_years >= 1, "early stopping needs --val-years >= 1"

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    local_dirs = [args.data, args.temperature_dir]
    include_lat = not args.no_lat
    use_anchor = not args.no_anchor

    def src(fname):
        return resolve_input_file(fname, local_dirs, args.github_base_url, args.cache_dir)

    map_grid_4ch = load_map_variables_grid(src(MAP_VARIABLES_FILE), epoch=args.epoch)
    static_spatial_feats = build_static_spatial_features(
        map_grid_4ch,
        kernel_size=args.kernel_size,
        vector_kernel_size=vec_k,
        include_water_vectors=not args.no_water_vectors,
        water_power=args.water_power,
        scale_ew=args.scale_ew_cos_lat,
        include_latitude=include_lat,
    )
    S = static_spatial_feats.shape[1]
    KK = args.kernel_size * args.kernel_size
    n_anchor_cols = 3 if use_anchor else 0
    use_extra = not args.no_extra_features
    base_dim = S + KK + n_anchor_cols            # old layout: [static | temp kernel | anchor cols]
    extra_static_feats = build_extra_static_features(map_grid_4ch, args.ray_max_steps) if use_extra else None
    feature_dim = base_dim + (N_EXTRA_STATIC + N_EXTRA_TEMP if use_extra else 0)   # extras are APPENDED
    cat_idx = [S + KK] if use_anchor else []

    T_hist_dict, ht = read_wide_temperature(src(HISTORICAL_TEMPERATURE_FILE))
    T_hist = T_hist_dict["-"]
    T_sc, st = read_wide_temperature(src(SCENARIO_TEMPERATURE_FILE), scenario_col=True)
    Kh, kh_years = read_koppen(src(HISTORICAL_KOPPEN_FILE))
    Kh_pix, Kh_str = Kh["-"]
    Ksc, ks_years = read_koppen(src(SCENARIO_KOPPEN_FILE), scenario_col=True)

    scenarios = sorted(set(T_sc) & set(Ksc))
    holdout_sc = [s for s in args.holdout_scenarios.split(",") if s]
    for s in holdout_sc:
        assert s in scenarios, f"unknown holdout scenario {s}"
    train_sc = scenarios if args.train_scenarios == "all" else args.train_scenarios.split(",")
    train_sc = [s for s in train_sc if s not in holdout_sc]
    assert args.bridge_scenario in T_sc

    common = sorted(set(ht) & set(st))
    if common:
        d = [float(np.nanmean(T_hist[Kh_pix, ht[y]] - T_sc[args.bridge_scenario][Kh_pix, st[y]])) for y in common]
        log(f"temperature seam: {len(common)} overlapping years, mean(hist - {args.bridge_scenario}) = {np.mean(d):+.3f} C")

    classes = np.unique(np.concatenate([np.unique(Kh_str)] + [np.unique(v[1]) for v in Ksc.values()]))
    classes = classes[classes != ""]
    n_classes = len(classes)
    enc = lambda arr: np.where(arr == "", -1, np.searchsorted(classes, arr)).astype(np.int16)
    Kh_code = enc(Kh_str)
    Ksc_code = {s: (p, enc(l)) for s, (p, l) in Ksc.items()}
    main_group = np.array([c[0] for c in classes])
    main_code = np.unique(main_group, return_inverse=True)[1]

    r_all, _ = np.divmod(np.arange(N_PIXELS), N_LON)
    heldout = np.zeros(N_PIXELS, dtype=bool)
    if args.holdout_frac > 0:
        hb = max(1, args.holdout_block_deg)
        lat_band_id = r_all // hb
        land_bands = np.unique(lat_band_id[Kh_pix])
        n_hold = max(1, int(round(args.holdout_frac * len(land_bands))))
        chosen = np.random.default_rng(SEED + 1).choice(land_bands, n_hold, replace=False)
        heldout = np.isin(lat_band_id, chosen)
        log(f"holding out {n_hold}/{len(land_bands)} latitude bands ({heldout[Kh_pix].mean():.1%} of land pixels)")

    def build_block(pix, T_full, K_rows, years, tcol, kcol, parity=None, with_prev=False, scn=None):
        ys = [y for y in years if y in tcol and y in kcol and (parity is None or y % 2 == parity)
              and (not with_prev or (y - 1) in kcol)]
        blk = {
            "pix": pix,
            "T": T_full,
            "tidx": np.array([tcol[y] for y in ys], dtype=np.int64),
            "K": K_rows[:, [kcol[y] for y in ys]],
            "ys": np.array(ys),
            "scn": scn,          # scenario whose temperature continues this block (None = historical/bridge)
        }
        if with_prev:  # label of the previous year = the "copy last year" baseline for held-out odd years
            blk["prevK"] = K_rows[:, [kcol[y - 1] for y in ys]]
        return blk

    train_blocks, test_blocks = [], {}
    par = 0 if even_mode else None          # even years only when --even-years
    train_blocks.append(("hist_1900_2014", False, build_block(Kh_pix, T_hist, Kh_code, range(1900, min(args.train_end, 2014) + 1), ht, kh_years, parity=par)))
    train_blocks.append(("bridge_2015_2025", False, build_block(Kh_pix, T_sc[args.bridge_scenario], Kh_code, range(2015, min(args.train_end, 2025) + 1), st, kh_years, parity=par)))
    for s in train_sc:
        p, kc = Ksc_code[s]
        train_blocks.append((f"scen_{s}_2026_{args.train_end}", True, build_block(p, T_sc[s], kc, range(2026, args.train_end + 1), st, ks_years, parity=par, scn=s)))
    if even_mode:
        # held-out ODD years: historical, bridge and every scenario; baseline = label of the previous (even) year
        test_blocks["hist_odd"] = build_block(Kh_pix, T_hist, Kh_code, range(1901, 2015), ht, kh_years, parity=1, with_prev=True)
        test_blocks["bridge_odd"] = build_block(Kh_pix, T_sc[args.bridge_scenario], Kh_code, range(2015, 2026), st, kh_years, parity=1, with_prev=True)
        for s in scenarios:
            p, kc = Ksc_code[s]
            test_blocks[s] = build_block(p, T_sc[s], kc, range(2026, args.test_end + 1), st, ks_years, parity=1, with_prev=True, scn=s)
    else:
        for s in scenarios:
            p, kc = Ksc_code[s]
            blk = build_block(p, T_sc[s], kc, range(args.train_end + 1, args.test_end + 1), st, ks_years, scn=s)
            anchor_K = np.full(N_PIXELS, -1, dtype=np.int16)
            if args.train_end >= 2026:
                anchor_K[p] = kc[:, ks_years[args.train_end]]
            else:
                anchor_K[Kh_pix] = Kh_code[:, kh_years[args.train_end]]
            blk["anchor_K_full"] = anchor_K
            blk["anchor_T_full"] = T_sc[s][:, st[args.train_end]]
            blk["anchor_year"] = args.train_end
            test_blocks[s] = blk

    def full_temperature_series(scn):
        """(N_PIXELS, 200) annual temperature for 1900..2099: historical before 2015, the bridge scenario
        for 2015-2025, then scenario `scn` (the bridge scenario when scn is None)."""
        arr = np.full((N_PIXELS, 200), np.nan, dtype=np.float32)
        for y, i in ht.items():
            if 1900 <= y < 2015:
                arr[:, y - 1900] = T_hist[:, i]
        for y, i in st.items():
            if 2015 <= y <= 2099:
                src = args.bridge_scenario if (y <= 2025 or scn is None) else scn
                arr[:, y - 1900] = T_sc[src][:, i]
        return arr

    def temp_kernel_for_grid(temp_1d_64800):
        t_grid = temp_1d_64800.reshape(N_LAT, N_LON, 1)
        return build_neighbourhood_features_for_every_pixel(t_grid, args.kernel_size).reshape(N_PIXELS, -1)

    def year_selection(n_yr, train):
        if train and args.year_stride > 1:
            return np.arange(n_yr - 1, -1, -args.year_stride)[::-1]
        return np.arange(n_yr)

    def block_cells(blk):
        if blk["K"].size == 0:
            return 0
        return int((~heldout[blk["pix"]]).sum()) * len(year_selection(blk["K"].shape[1], True))

    def build_rows(blk, frac=1.0, train=True):
        pix, Tfull, tidx, K, ys = blk["pix"], blk["T"], blk["tidx"], blk["K"], blk["ys"]
        yr_sel = year_selection(K.shape[1], train)
        pix_ok = np.flatnonzero(~heldout[pix]) if train else np.arange(len(pix))
        n_sel = len(yr_sel)
        total_cells = len(pix_ok) * n_sel
        flat = np.arange(total_cells) if frac >= 1.0 else rng.choice(total_cells, int(total_cells * frac), replace=False)
        a, b = np.divmod(flat, n_sel)
        pi, yi = pix_ok[a], yr_sel[b]
        y_code = K[pi, yi]
        gp = pix[pi]
        keep = y_code >= 0

        if train:
            lag = rng.integers(1, args.max_lag + 1, size=len(pi))
            ai = np.maximum(yi - lag, 0)
            lag_eff = (yi - ai).astype(np.float32)
            a_code = K[pi, ai]
            a_T = Tfull[gp, tidx[ai]]
            keep &= a_code >= 0
            if use_anchor:
                keep &= lag_eff > 0
        elif "prevK" in blk:               # even-years mode: baseline = previous year's label
            a_code = blk["prevK"][pi, yi]
            a_T = np.zeros(len(pi), dtype=np.float32)
            lag_eff = np.ones(len(pi), dtype=np.float32)
            keep &= a_code >= 0
        else:
            a_code = blk["anchor_K_full"][gp]
            a_T = blk["anchor_T_full"][gp]
            lag_eff = (ys[yi] - blk["anchor_year"]).astype(np.float32)
            keep &= a_code >= 0

        pi, yi, gp = pi[keep], yi[keep], gp[keep]
        y_code, a_code, a_T, lag_eff = y_code[keep], a_code[keep], a_T[keep], lag_eff[keep]
        local_t = Tfull[gp, tidx[yi]].astype(np.float32)

        X = np.empty((len(pi), feature_dim), dtype=np.float32)
        X[:, :S] = static_spatial_feats[gp]
        for y_idx in np.unique(yi):
            m = yi == y_idx
            t_year = Tfull[:, tidx[y_idx]]
            X[m, S : S + KK] = temp_kernel_for_grid(t_year)[gp[m]]
            if use_extra:   # local temperature minus the zonal (same-latitude) mean for that year
                zonal = np.nanmean(t_year.reshape(N_LAT, N_LON), axis=1)
                X[m, base_dim + N_EXTRA_STATIC + 3] = t_year[gp[m]] - zonal[r_all[gp[m]]]
        if use_anchor:
            X[:, S + KK] = a_code
            X[:, S + KK + 1] = lag_eff
            X[:, S + KK + 2] = local_t - a_T
        if use_extra:
            X[:, base_dim : base_dim + N_EXTRA_STATIC] = extra_static_feats[gp]
            series = full_temperature_series(blk["scn"])
            col = (ys[yi] - 1900).astype(np.int64)
            tm10 = trailing_mean(series, 10)[gp, col]
            tm30 = trailing_mean(series, 30)[gp, col]
            del series
            X[:, base_dim + N_EXTRA_STATIC] = tm10
            X[:, base_dim + N_EXTRA_STATIC + 1] = tm30
            X[:, base_dim + N_EXTRA_STATIC + 2] = tm10 - tm30
        w = np.where(a_code != y_code, args.change_weight, 1.0).astype(np.float32)
        return {"X": X, "y": y_code.astype(np.int32), "yrs": ys[yi], "pix": gp, "anchor": a_code.astype(np.int16), "w": w}

    active = [(n, sc, b) for n, sc, b in train_blocks if b["K"].size]
    cells = [block_cells(b) for _, _, b in active]
    total_possible = sum(cells)
    frac = 1.0 if not args.max_train_rows else min(1.0, args.max_train_rows / max(total_possible, 1))
    est_rows = int(total_possible * frac)
    log(f"~{est_rows:,} training rows x {feature_dim} features = ~{est_rows * feature_dim * 4 / 1e9:.1f} GB float32 "
        f"(lat={include_lat}, lon=False); peak memory is roughly twice that while the blocks are joined")

    # ------------------------------ build training / validation rows ------------------------------
    parts_tr, parts_va = [], []
    for name, is_scen, blk in (tqdm(active, desc="building training rows", unit="block") if tqdm else active):
        d = build_rows(blk, frac=frac, train=True)
        if even_mode:
            # validate like the test: a spread of held-out EVEN years from every period and scenario. (Validating only
            # on the last scenario years punishes the model for correct historical fitting and stops training early.)
            vm = (d["yrs"] // 2) % args.val_every == 0
        elif is_scen:
            vm = d["yrs"] > d["yrs"].max() - args.val_years
        else:
            vm = None
        if vm is not None:
            parts_va.append({k: v[vm] for k, v in d.items()})
            d = {k: v[~vm] for k, v in d.items()}
        parts_tr.append(d)
    assert parts_va and sum(len(p["y"]) for p in parts_va) > 0, \
        "no validation rows: need at least one training scenario and --val-years >= 1"

    anchor_agree = float(np.mean(np.concatenate([p["anchor"] == p["y"] for p in parts_tr]))) if use_anchor else None
    Xtr = np.concatenate([p["X"] for p in parts_tr])
    ytr = np.concatenate([p["y"] for p in parts_tr])
    wtr = np.concatenate([p["w"] for p in parts_tr])
    del parts_tr
    total_train_rows = len(ytr)
    class_counts = np.bincount(ytr, minlength=n_classes)
    log(f"training rows: {total_train_rows:,} | feature dim: {feature_dim}")
    if use_anchor:
        log(f"sanity: copying the anchor is right for {anchor_agree:.1%} of training rows "
            "(historical data: ~93-99%); the trained model should reach about this or higher")

    Xva = np.concatenate([p["X"] for p in parts_va])
    yva = np.concatenate([p["y"] for p in parts_va])
    del parts_va
    if len(yva) > 300_000:
        sel = rng.choice(len(yva), 300_000, replace=False)
        Xva, yva = Xva[sel], yva[sel]

    # ------------------------------ train (single pass, early stopping) ------------------------------
    params = dict(
        objective="multiclass",
        num_class=n_classes,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=0.8,
        seed=SEED,
        num_threads=0,
        verbosity=-1,
    )
    cat_arg = cat_idx if cat_idx else "auto"
    params.update(min_data_in_leaf=args.min_data_in_leaf, lambda_l2=args.lambda_l2)

    def fit(device):
        p = dict(params)
        if device != "cpu":
            p["device_type"] = device
        dtrain = lgb.Dataset(Xtr, ytr, weight=wtr, categorical_feature=cat_arg)
        dval = lgb.Dataset(Xva, yva, reference=dtrain, categorical_feature=cat_arg)
        progress = TrainProgress(args.n_estimators, desc=f"training ({device})")
        try:
            return lgb.train(
                p, dtrain, num_boost_round=args.n_estimators, valid_sets=[dval],
                callbacks=[progress, lgb.early_stopping(args.early_stopping_rounds, verbose=True)],
            )
        finally:
            progress.close()

    try:
        booster = fit(args.device)
    except Exception as e:  # noqa: BLE001 - LightGBM raises LightGBMError for missing GPU build / OpenCL / CUDA
        if args.device == "cpu":
            raise
        log(f"GPU training unavailable ({str(e).strip().splitlines()[0][:150]}); falling back to CPU. "
            "The pip wheel supports --device gpu (needs an OpenCL driver); --device cuda needs a CUDA build of LightGBM.")
        booster = fit("cpu")
    best_iter = getattr(booster, "best_iteration", None) or None
    log(f"trained {booster.current_iteration()} rounds; best iteration = {best_iter}")
    if best_iter and best_iter < 100:
        log("WARNING: early stopping picked fewer than 100 rounds, so the model is probably under-fitted. "
            "Check that the validation years resemble the test years, or raise --early-stopping-rounds.")
    if even_mode and args.holdout_frac == 0:
        log("note: odd test years are interleaved with training years at the SAME pixels, so this accuracy is optimistic "
            "for new places. Add --holdout-frac 0.2 to measure generalisation to unseen latitude bands.")

    # sanity check: does the model beat "copy the anchor" on a sample of its own training rows?
    sel = rng.choice(len(ytr), min(200_000, len(ytr)), replace=False)
    train_sample_acc = float((predict_codes(booster, Xtr[sel]) == ytr[sel]).mean())
    log(f"training-sample accuracy {train_sample_acc:.1%}" + (f" vs copy-the-anchor {anchor_agree:.1%}" if use_anchor else ""))
    if use_anchor and train_sample_acc < anchor_agree - 0.05:
        log("WARNING: the model is clearly WORSE than copying the anchor even on its own training data. "
            "It is under-trained: raise --n-estimators / --max-train-rows or --learning-rate.")
    del Xtr, ytr, Xva, yva

    majority = int(class_counts.argmax())

    def summarize(y, p, a, mask=None):
        if mask is not None:
            y, p, a = y[mask], p[mask], a[mask]
        n = len(y)
        if n == 0:
            return None
        ch = y != a
        acc, pers = float((p == y).mean()), float((a == y).mean())
        return {
            "rows": int(n),
            "accuracy": acc,
            "accuracy_main_group": float((main_code[p] == main_code[y]).mean()),
            "persistence_accuracy": pers,
            "persistence_main_group": float((main_code[a] == main_code[y]).mean()),
            "majority_class_baseline": float((y == majority).mean()),
            "frac_rows_changed_vs_anchor": float(ch.mean()),
            "accuracy_on_changed_rows": float((p[ch] == y[ch]).mean()) if ch.any() else None,
            "accuracy_on_unchanged_rows": float((p[~ch] == y[~ch]).mean()) if (~ch).any() else None,
            "skill_vs_persistence": (acc - pers) / (1 - pers) if pers < 1 else None,
        }

    metrics = {
        "epoch": args.epoch,
        "kernel_size": args.kernel_size,
        "train_rows": int(total_train_rows),
        "feature_dim": int(feature_dim),
        "includes_latitude": include_lat,
        "includes_longitude": False,
        "use_anchor": use_anchor,
        "extra_features": use_extra,
        "train_end": args.train_end,
        "boosting_rounds": int(booster.current_iteration()),
        "best_iteration": best_iter,
        "train_anchor_agreement": anchor_agree,
        "train_sample_accuracy": train_sample_acc,
        "heldout_scenarios": holdout_sc,
        "heldout_region_pixel_fraction": float(heldout[Kh_pix].mean()) if args.holdout_frac > 0 else 0.0,
        "scenarios": {},
    }
    pooled = {"y": [], "p": [], "a": []}

    for s, blk in test_blocks.items():
        if blk["K"].size == 0:
            continue
        d = build_rows(blk, frac=1.0, train=False)
        if len(d["y"]) == 0:
            continue
        pred = predict_codes(booster, d["X"])
        y, a = d["y"].astype(np.int16), d["anchor"]
        res = summarize(y, pred, a)
        res["scenario_seen_in_training"] = even_mode or s in train_sc
        edges = np.array_split(np.unique(d["yrs"]), 3)
        res["by_period"] = {f"{e[0]}-{e[-1]}": summarize(y, pred, a, np.isin(d["yrs"], e)) for e in edges if len(e)}
        if args.holdout_frac > 0:
            ho = heldout[d["pix"]]
            res["seen_regions"] = summarize(y, pred, a, ~ho)
            res["heldout_regions"] = summarize(y, pred, a, ho)
        metrics["scenarios"][s] = res
        pooled["y"].append(y), pooled["p"].append(pred), pooled["a"].append(a)
        log(f"{s}{'' if res['scenario_seen_in_training'] else ' (UNSEEN)'}: acc={res['accuracy']:.4f} persistence={res['persistence_accuracy']:.4f} acc_on_changed={res['accuracy_on_changed_rows']}")
        del d

    if pooled["y"]:
        y, p, a = (np.concatenate(pooled[k]) for k in ("y", "p", "a"))
        metrics["pooled"] = summarize(y, p, a)
        conf = np.bincount(y.astype(np.int64) * n_classes + p, minlength=n_classes**2).reshape(n_classes, n_classes)
        support, predicted, tp = conf.sum(1), conf.sum(0), np.diag(conf)
        pers_tp = np.bincount(y[a == y], minlength=n_classes)
        per_class = {}
        for k, c in enumerate(classes):
            prec = tp[k] / predicted[k] if predicted[k] else None
            rec = tp[k] / support[k] if support[k] else None
            f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
            per_class[str(c)] = {
                "support": int(support[k]),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "persistence_recall": float(pers_tp[k] / support[k]) if support[k] else None,
            }
        metrics["per_class"] = per_class
        pm = metrics["pooled"]
        log(f"POOLED test: model {pm['accuracy']:.4f} vs persistence {pm['persistence_accuracy']:.4f}")

    tag = f"{args.epoch}_k{args.kernel_size}_{'lat' if include_lat else 'nolat'}_{'anchor' if use_anchor else 'noanchor'}_1900_{args.train_end}{'_evenyears' if even_mode else ''}{'_x' if use_extra else ''}"
    joblib.dump(
        {
            "booster": booster,
            "classes": [str(c) for c in classes],
            "epoch": args.epoch,
            "kernel_size": args.kernel_size,
            "vector_kernel_size": vec_k,
            "include_water_vectors": not args.no_water_vectors,
            "water_power": args.water_power,
            "scale_ew_cos_lat": args.scale_ew_cos_lat,
            "max_lag": args.max_lag,
            "includes_latitude": include_lat,
            "includes_longitude": False,
            "use_anchor": use_anchor,
            "extra_features": use_extra,
            "ray_max_steps": args.ray_max_steps,
            "n_extra_static": N_EXTRA_STATIC if use_extra else 0,
            "n_extra_temp": N_EXTRA_TEMP if use_extra else 0,
            "feature_dim": feature_dim,
            "best_iteration": best_iter,
        },
        os.path.join(args.out, f"koppen_{tag}.joblib"),
        compress=3,
    )
    booster.save_model(os.path.join(args.out, f"koppen_{tag}.txt"))
    with open(os.path.join(args.out, f"metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    model_path = os.path.join(args.out, f"koppen_{tag}.joblib")
    log("done")
    log(f"model saved: {model_path}")
    if use_anchor:
        log("note: trained WITH the anchor feature, so predictions also need anchor_koppen / anchor_temperature / years_since_anchor. "
            "Retrain with --no-anchor for a model that needs only map variables + temperature.")
    log(f"use it:  from koppen_predict import load_model; predict = load_model(r'{model_path}'); "
        "koppen = predict(map_vars, temperature)")


if __name__ == "__main__":
    main()
