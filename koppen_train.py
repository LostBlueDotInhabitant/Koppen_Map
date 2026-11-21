#!/usr/bin/env python3
"""
Merged Köppen pipeline combining:
  * kernelize-style spatial/vector/latitude features, 0ka/21ka map epochs, & optional GitHub/local sources
  * LightGBM multiclass training (native lgb.train), blockwise/stride/row controls, & persistence/change metrics

Training can run on the GPU (--device auto|gpu|cuda|cpu). Row/feature building always runs on the CPU;
only LightGBM's boosting uses the GPU. The saved model is device-independent and predicts on CPU.

Constraints enforced:
  * Pixel latitude (lat) is included as a feature (when --no-lat is not passed).
  * Longitude (lon) is NEVER included as a model feature or spatial hold-out axis.
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
DEFAULT_KERNEL_SIZE = 11
ALL_SCENARIOS = ("ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585")
DEFAULT_KOPPEN_PERIODS = [
    (1900, 1900, 1930),
    (1931, 1931, 1960),
    (1961, 1961, 1990),
    (1991, 1991, 2020),
]
SEED = 42


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


class StepBar:
    """Small progress bar for the feature-building ('kernelization') stages. Uses tqdm if installed."""

    def __init__(self, total, desc, unit="step", leave=True):
        self.total = max(1, int(total))
        self.n = 0
        self.desc = desc
        self.leave = leave
        self.last_print = 0.0
        self.t0 = time.time()
        self.bar = (
            tqdm(total=self.total, desc=desc, unit=unit, leave=leave, dynamic_ncols=True)
            if tqdm is not None else None
        )

    def set_label(self, label):
        self.desc = label
        if self.bar is not None:
            self.bar.set_description(label)
        else:
            self._print(force=True)

    def update(self, n=1):
        self.n += n
        if self.bar is not None:
            self.bar.update(n)
        else:
            self._print()

    def _print(self, force=False):
        now = time.time()
        if force or now - self.last_print > 0.25 or self.n >= self.total:
            self.last_print = now
            frac = min(1.0, self.n / self.total)
            fill = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
            print(f"\r{self.desc[:48]:<48} {frac:6.1%} [{fill}] {self.n}/{self.total}", end="", flush=True)

    def close(self):
        if self.bar is not None:
            self.bar.close()
        elif self.leave:
            print(flush=True)
        else:
            print("\r" + " " * 110 + "\r", end="", flush=True)


class TrainProgress:
    """LightGBM callback: one progress bar over all boosting rounds (works across blockwise passes)."""

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

    def set_desc(self, desc):
        self.desc = desc
        if self.bar is not None:
            self.bar.set_description(desc)

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
    progress=True,
):
    """Build per-pixel static features: map kernel + optional 12 water/ice vectors + optional latitude (no longitude)."""
    land_frac = map_grid_4ch[..., 2]
    ice_frac = map_grid_4ch[..., 3]

    def vectors(marked):
        return compute_vectors_toward_marked_pixels(
            marked, vector_kernel_size, water_power, scale_ew
        ).reshape(N_PIXELS, -1)

    steps = [("map neighbourhood kernel",
              lambda: build_neighbourhood_features_for_every_pixel(map_grid_4ch, kernel_size).reshape(N_PIXELS, -1))]
    if include_water_vectors:
        steps += [
            ("vectors toward water", lambda: vectors(land_frac == 0)),
            ("vectors toward open water", lambda: vectors((land_frac == 0) & (ice_frac == 0))),
            ("vectors toward ice", lambda: vectors(ice_frac > 0)),
        ]
    if include_latitude:
        steps.append(("latitude", lambda: np.repeat(np.arange(N_LAT, dtype=np.float32) - 89.5, N_LON).reshape(N_PIXELS, 1)))

    bar = StepBar(len(steps), "kernelizing map", unit="step") if progress else None
    parts = []
    for label, fn in steps:
        if bar:
            bar.set_label(f"kernelizing map: {label}")
        parts.append(fn())
        if bar:
            bar.update()
    if bar:
        bar.close()
    return np.hstack(parts).astype(np.float32)


def compute_area_weighted_global_mean(grid_2d):
    """Area-weighted global mean of a (180, 360) grid."""
    weights = np.cos(np.deg2rad(np.arange(N_LAT) - 89.5))[:, None]
    return float((grid_2d * weights).sum() / (weights.sum() * grid_2d.shape[1]))


def compute_period_mean_temperature_grids(temp_rows_or_dict, tcol_or_periods, periods=DEFAULT_KOPPEN_PERIODS):
    """Average yearly grids over each (label_year, first_year, last_year) period."""
    out = []
    if isinstance(temp_rows_or_dict, dict):
        for label_year, first_y, last_y in periods:
            mean_g = np.mean([temp_rows_or_dict[y] for y in range(first_y, last_y + 1)], axis=0).astype(np.float32)
            out.append((label_year, mean_g))
        return out
    T_full, tcol = temp_rows_or_dict, tcol_or_periods
    for label_year, first_y, last_y in periods:
        ys = [y for y in range(first_y, last_y + 1) if y in tcol]
        mean_g = np.mean([T_full[:, tcol[y]].reshape(N_LAT, N_LON) for y in ys], axis=0).astype(np.float32)
        out.append((label_year, mean_g))
    return out


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


def pick_device(requested, gpu_params):
    """Return 'cuda', 'gpu' (OpenCL) or 'cpu'. 'auto' tries cuda -> gpu -> cpu and falls back quietly;
    an explicitly requested GPU device that does not work stops with a clear message instead."""
    candidates = ["cuda", "gpu", "cpu"] if requested == "auto" else [requested]
    # tiny multiclass problem with a categorical column + weights, like the real training
    r = np.random.default_rng(0)
    X = r.random((2000, 6)).astype(np.float32)
    X[:, 0] = r.integers(0, 4, 2000)
    y = r.integers(0, 3, 2000)
    for dev in candidates:
        if dev == "cpu":
            return "cpu"
        try:
            lgb.train(
                dict(objective="multiclass", num_class=3, device_type=dev, verbosity=-1, **gpu_params),
                lgb.Dataset(X, y, weight=np.ones(2000, np.float32), categorical_feature=[0]),
                num_boost_round=2,
            )
            return dev
        except Exception as e:  # LightGBM raises LightGBMError when the build has no such device
            first = (str(e).strip().splitlines() or ["unknown error"])[0][:160]
            log(f"device '{dev}' not usable: {first}")
            if requested != "auto":
                raise SystemExit(
                    f"--device {requested} failed (your LightGBM build probably has no GPU support). "
                    "Use --device auto to fall back to the CPU automatically, or --device cpu."
                )
    return "cpu"


def predict_codes(booster, X, chunk=400_000):
    out = np.empty(len(X), dtype=np.int16)
    for i in range(0, len(X), chunk):
        out[i : i + chunk] = booster.predict(X[i : i + chunk]).argmax(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="Directory containing local CSVs")
    ap.add_argument("--temperature-dir", default=None, help="Optional separate directory for temperature CSVs")
    ap.add_argument("--github-base-url", default=None, help="Optional raw GitHub base URL fallback")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--out", required=True, help="Output directory for model and metrics")
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
    ap.add_argument("--device", choices=["auto", "gpu", "cuda", "cpu"], default="auto",
                    help="auto = try cuda, then OpenCL gpu, then cpu. 'cuda' = NVIDIA CUDA build, 'gpu' = OpenCL build")
    ap.add_argument("--gpu-platform-id", type=int, default=None, help="OpenCL platform id (only for --device gpu)")
    ap.add_argument("--gpu-device-id", type=int, default=None, help="which GPU to use, if you have several")
    ap.add_argument("--n-estimators", type=int, default=250)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-train-rows", type=int, default=0, help="0 = use every selected row")
    ap.add_argument("--year-stride", type=int, default=1)
    ap.add_argument("--blockwise-fit", action="store_true")
    ap.add_argument("--blockwise-passes", type=int, default=3)
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--max-lag", type=int, default=49)
    ap.add_argument("--change-weight", type=float, default=1.0)
    ap.add_argument("--holdout-frac", type=float, default=0.0, help="Fraction of latitude bands held out")
    ap.add_argument("--holdout-block-deg", type=int, default=30)
    ap.add_argument("--holdout-scenarios", default="")
    ap.add_argument("--val-years", type=int, default=5)
    ap.add_argument("--early-stopping-rounds", type=int, default=30)
    args = ap.parse_args()

    assert args.kernel_size % 2 == 1, "kernel_size must be odd"
    vec_k = args.kernel_size if args.vector_kernel_size is None else args.vector_kernel_size
    assert vec_k % 2 == 1, "vector_kernel_size must be odd"
    assert args.train_end >= 2025, "train_end must be >= 2025"
    assert args.year_stride >= 1

    gpu_params = {}
    if args.gpu_platform_id is not None:
        gpu_params["gpu_platform_id"] = args.gpu_platform_id
    if args.gpu_device_id is not None:
        gpu_params["gpu_device_id"] = args.gpu_device_id
    device = pick_device(args.device, gpu_params)
    log(f"training device: {device.upper()}" + ("" if device != "cpu" else " (no usable GPU build found)" if args.device == "auto" else ""))

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    local_dirs = [args.data, args.temperature_dir]
    include_lat = not args.no_lat
    use_anchor = not args.no_anchor

    def src(fname):
        return resolve_input_file(fname, local_dirs, args.github_base_url, args.cache_dir)

    map_grid_4ch = load_map_variables_grid(src("MapVariables.csv"), epoch=args.epoch)
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
    feature_dim = S + KK + n_anchor_cols
    cat_idx = [S + KK] if use_anchor else []

    T_hist_dict, ht = read_wide_temperature(src("tas_annual_mean_1deg_wide_celsius.csv"))
    T_hist = T_hist_dict["-"]
    T_sc, st = read_wide_temperature(src("temp_annual_all_scenarios_2015_2099.csv"), scenario_col=True)
    Kh, kh_years = read_koppen(src("koppen_historical_1900_2026.csv"))
    Kh_pix, Kh_str = Kh["-"]
    Ksc, ks_years = read_koppen(src("koppen_scenarios_2026_2100.csv"), scenario_col=True)

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

    def build_block(pix, T_full, K_rows, years, tcol, kcol):
        ys = [y for y in years if y in tcol and y in kcol]
        return {
            "pix": pix,
            "T": T_full,
            "tidx": np.array([tcol[y] for y in ys], dtype=np.int64),
            "K": K_rows[:, [kcol[y] for y in ys]],
            "ys": np.array(ys),
        }

    train_blocks, test_blocks = [], {}
    train_blocks.append(("hist_1900_2014", False, build_block(Kh_pix, T_hist, Kh_code, range(1900, min(args.train_end, 2014) + 1), ht, kh_years)))
    train_blocks.append(("bridge_2015_2025", False, build_block(Kh_pix, T_sc[args.bridge_scenario], Kh_code, range(2015, min(args.train_end, 2025) + 1), st, kh_years)))
    for s in train_sc:
        p, kc = Ksc_code[s]
        train_blocks.append((f"scen_{s}_2026_{args.train_end}", True, build_block(p, T_sc[s], kc, range(2026, args.train_end + 1), st, ks_years)))
    for s in scenarios:
        p, kc = Ksc_code[s]
        blk = build_block(p, T_sc[s], kc, range(args.train_end + 1, args.test_end + 1), st, ks_years)
        anchor_K = np.full(N_PIXELS, -1, dtype=np.int16)
        if args.train_end >= 2026:
            anchor_K[p] = kc[:, ks_years[args.train_end]]
        else:
            anchor_K[Kh_pix] = Kh_code[:, kh_years[args.train_end]]
        blk["anchor_K_full"] = anchor_K
        blk["anchor_T_full"] = T_sc[s][:, st[args.train_end]]
        blk["anchor_year"] = args.train_end
        test_blocks[s] = blk

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

    def build_rows(blk, frac=1.0, train=True, desc="rows"):
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
        uniq_years = np.unique(yi)
        kbar = StepBar(len(uniq_years), f"kernelizing temperature [{desc}]", unit="year", leave=False)
        for y_idx in uniq_years:
            m = yi == y_idx
            X[m, S : S + KK] = temp_kernel_for_grid(Tfull[:, tidx[y_idx]])[gp[m]]
            kbar.update()
        kbar.close()
        if use_anchor:
            X[:, S + KK] = a_code
            X[:, S + KK + 1] = lag_eff
            X[:, S + KK + 2] = local_t - a_T
        w = np.where(a_code != y_code, args.change_weight, 1.0).astype(np.float32)
        return {"X": X, "y": y_code.astype(np.int32), "yrs": ys[yi], "pix": gp, "anchor": a_code.astype(np.int16), "w": w}

    active = [(n, sc, b) for n, sc, b in train_blocks if b["K"].size]
    cells = [block_cells(b) for _, _, b in active]
    total_possible = sum(cells)
    frac = 1.0 if not args.max_train_rows else min(1.0, args.max_train_rows / max(total_possible, 1))
    est_rows = int(total_possible * frac)
    log(f"~{est_rows:,} training rows x {feature_dim} features = ~{est_rows * feature_dim * 4 / 1e9:.1f} GB float32 (lat={include_lat}, lon=False)")

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
        device_type=device,
        **gpu_params,
    )
    cat_arg = cat_idx if cat_idx else "auto"
    class_counts = np.zeros(n_classes, dtype=np.int64)
    total_train_rows = 0
    best_iter = None

    if args.blockwise_fit:
        passes = max(1, args.blockwise_passes)
        per_block = [max(1, round(args.n_estimators * (c / max(total_possible, 1)) / passes)) for c in cells]
        booster = None
        progress = TrainProgress(passes * sum(per_block), desc="training")
        for p_i in range(passes):
            for k in rng.permutation(len(active)):
                name, _, blk = active[k]
                d = build_rows(blk, frac=frac, train=True, desc=f"pass {p_i + 1}, {name}")
                if len(d["y"]) == 0:
                    continue
                if p_i == 0:
                    total_train_rows += len(d["y"])
                    class_counts += np.bincount(d["y"], minlength=n_classes)
                progress.set_desc(f"pass {p_i + 1}/{passes} '{name}'")
                ds = lgb.Dataset(d["X"], d["y"], weight=d["w"], categorical_feature=cat_arg, free_raw_data=False)
                booster = lgb.train(params, ds, num_boost_round=per_block[k], init_model=booster, callbacks=[progress])
                del d, ds
        progress.close()
        assert booster is not None, "no training rows"
    else:
        parts_tr, parts_va = [], []
        for name, is_scen, blk in (tqdm(active, desc="building training rows", unit="block") if tqdm else active):
            d = build_rows(blk, frac=frac, train=True, desc=name)
            if args.val_years > 0 and is_scen:
                vm = d["yrs"] > args.train_end - args.val_years
                parts_va.append({k: v[vm] for k, v in d.items()})
                d = {k: v[~vm] for k, v in d.items()}
            parts_tr.append(d)
        Xtr = np.concatenate([p["X"] for p in parts_tr])
        ytr = np.concatenate([p["y"] for p in parts_tr])
        wtr = np.concatenate([p["w"] for p in parts_tr])
        del parts_tr
        total_train_rows = len(ytr)
        class_counts = np.bincount(ytr, minlength=n_classes)
        log(f"training rows: {total_train_rows:,} | feature dim: {feature_dim}")
        dtrain = lgb.Dataset(Xtr, ytr, weight=wtr, categorical_feature=cat_arg)
        progress = TrainProgress(args.n_estimators, desc="training")
        valid_sets, callbacks = [], [progress]
        if parts_va:
            Xva = np.concatenate([p["X"] for p in parts_va])
            yva = np.concatenate([p["y"] for p in parts_va])
            if len(yva) > 300_000:
                sel = rng.choice(len(yva), 300_000, replace=False)
                Xva, yva = Xva[sel], yva[sel]
            valid_sets = [lgb.Dataset(Xva, yva, reference=dtrain, categorical_feature=cat_arg)]
            callbacks.append(lgb.early_stopping(args.early_stopping_rounds, verbose=True))
        booster = lgb.train(params, dtrain, num_boost_round=args.n_estimators, valid_sets=valid_sets or None, callbacks=callbacks)
        progress.close()
        best_iter = getattr(booster, "best_iteration", None) or None
        del Xtr, ytr, dtrain

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
        "device": device,
        "kernel_size": args.kernel_size,
        "train_rows": int(total_train_rows),
        "feature_dim": int(feature_dim),
        "includes_latitude": include_lat,
        "includes_longitude": False,
        "use_anchor": use_anchor,
        "train_end": args.train_end,
        "heldout_scenarios": holdout_sc,
        "heldout_region_pixel_fraction": float(heldout[Kh_pix].mean()) if args.holdout_frac > 0 else 0.0,
        "scenarios": {},
    }
    pooled = {"y": [], "p": [], "a": []}

    for s, blk in test_blocks.items():
        if blk["K"].size == 0:
            continue
        d = build_rows(blk, frac=1.0, train=False, desc=f"test {s}")
        if len(d["y"]) == 0:
            continue
        pred = predict_codes(booster, d["X"])
        y, a = d["y"].astype(np.int16), d["anchor"]
        res = summarize(y, pred, a)
        res["scenario_seen_in_training"] = s in train_sc
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

    tag = f"{args.epoch}_k{args.kernel_size}_{'lat' if include_lat else 'nolat'}_{'anchor' if use_anchor else 'noanchor'}_1900_{args.train_end}"
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