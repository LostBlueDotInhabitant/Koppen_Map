#!/usr/bin/env python3
"""
Ready-to-use Köppen predictor for models trained with koppen_train.py.

    from koppen_predict import load_model

    predict = load_model("out/koppen_0ka_k11_lat_noanchor_1900_2050.joblib")
    koppen = predict(map_vars, temperature)          # (180, 360) array of strings like "Cfb"

Inputs
------
map_vars     (180, 360, 4) float array with channels
             [altitude_m, ice_thickness_m, land_area_fraction_pct, ice_area_fraction_pct],
             OR a DataFrame in MapVariables.csv format (columns lat, lon, altitude_0ka_m, ...),
             OR a DataFrame with plain columns altitude_m, ice_thickness_m,
             land_area_fraction_pct, ice_area_fraction_pct (plus lat, lon),
             OR a dict {column name: (180, 360) array} using either naming style.
temperature  (180, 360) annual mean temperature in deg C, OR a (n, 180, 360) stack of years,
             OR a DataFrame with lat, lon and one value column per year / time step.

Grid convention: row 0 = latitude -89.5 (south), column 0 = longitude -179.5 (1-degree pixels).
Longitude is never used by the model; it only places pixels on the grid.

Models trained WITH the anchor (the default of koppen_train.py) additionally need
`anchor_koppen`, `anchor_temperature` and `years_since_anchor`. Models trained with
--no-anchor need only map_vars and temperature.
"""

import argparse
import hashlib
import os

import joblib
import numpy as np
import pandas as pd

from koppen_train import (
    N_LAT,
    N_LON,
    N_PIXELS,
    StepBar,
    build_neighbourhood_features_for_every_pixel,
    build_static_spatial_features,
    latlon_to_grid_indices,
    year_cols,
)

NON_VALUE_COLUMNS = {"lat", "lon", "src_lat", "src_lon", "scenario", "pixel_id"}
PLAIN_MAP_COLUMNS = ("altitude_m", "ice_thickness_m", "land_area_fraction_pct", "ice_area_fraction_pct")


# --------------------------------------------------------------------------------------
# Input conversion helpers
# --------------------------------------------------------------------------------------
def grid_from_table(df, column, fill=np.nan, dtype=np.float32):
    """DataFrame with lat/lon + `column` -> (180, 360) grid (row 0 = -89.5, col 0 = -179.5)."""
    r, c = latlon_to_grid_indices(df["lat"].to_numpy(), df["lon"].to_numpy())
    grid = np.full((N_LAT, N_LON), fill, dtype=dtype)
    grid[r, c] = df[column].to_numpy(dtype)
    return grid


def grid_to_dataframe(grid, name="koppen"):
    """(180, 360) grid -> long DataFrame of lat, lon (-179.5..179.5), value; empty cells dropped."""
    lat = np.repeat(np.arange(N_LAT) - 89.5, N_LON)
    lon = np.tile(np.arange(N_LON) - 179.5, N_LAT)
    flat = np.asarray(grid).reshape(N_PIXELS)
    keep = flat != "" if flat.dtype.kind in "US" else ~np.isnan(flat)
    return pd.DataFrame({"lat": lat[keep], "lon": lon[keep], name: flat[keep]})


def _to_map_grid(map_vars, epoch):
    if isinstance(map_vars, pd.DataFrame):
        epoch_cols = [f"altitude_{epoch}_m", f"ice_thickness_{epoch}_m",
                      f"land_area_fraction_{epoch}_pct", f"ice_area_fraction_{epoch}_pct"]
        cols = epoch_cols if all(c in map_vars for c in epoch_cols) else list(PLAIN_MAP_COLUMNS)
        missing = [c for c in cols if c not in map_vars]
        if missing:
            raise ValueError(f"map_vars DataFrame is missing columns {missing}")
        grid = np.stack([grid_from_table(map_vars, c) for c in cols], axis=-1)
    elif isinstance(map_vars, dict):
        epoch_cols = [f"altitude_{epoch}_m", f"ice_thickness_{epoch}_m",
                      f"land_area_fraction_{epoch}_pct", f"ice_area_fraction_{epoch}_pct"]
        cols = epoch_cols if all(c in map_vars for c in epoch_cols) else list(PLAIN_MAP_COLUMNS)
        missing = [c for c in cols if c not in map_vars]
        if missing:
            raise ValueError(f"map_vars dict is missing keys {missing}")
        grid = np.stack([np.asarray(map_vars[c], dtype=np.float32) for c in cols], axis=-1)
    else:
        grid = np.asarray(map_vars, dtype=np.float32)
    if grid.shape != (N_LAT, N_LON, 4):
        raise ValueError(f"map_vars must resolve to shape ({N_LAT}, {N_LON}, 4); got {grid.shape}")
    if np.isnan(grid).any():
        raise ValueError("map_vars has missing pixels (NaN); the full 180x360 grid is required")
    return np.ascontiguousarray(grid, dtype=np.float32)


def _to_temp_grids(temperature):
    """-> ((n, 180, 360) float32, was_single_2d_grid)."""
    if isinstance(temperature, pd.DataFrame):
        value_cols = [c for c in temperature.columns if c not in NON_VALUE_COLUMNS]
        if not value_cols:
            raise ValueError("temperature DataFrame needs at least one value column besides lat/lon")
        if all(str(c).isdigit() for c in value_cols):  # wide year columns -> chronological order
            value_cols = [year_cols(value_cols)[y] for y in sorted(year_cols(value_cols))]
        grids = np.stack([grid_from_table(temperature, c) for c in value_cols])
        return grids, len(value_cols) == 1
    arr = np.asarray(temperature, dtype=np.float32)
    if arr.shape == (N_LAT, N_LON):
        return arr[None], True
    if arr.ndim == 3 and arr.shape[1:] == (N_LAT, N_LON):
        return arr, False
    raise ValueError(f"temperature must be ({N_LAT}, {N_LON}) or (n, {N_LAT}, {N_LON}); got {arr.shape}")


# --------------------------------------------------------------------------------------
# The predictor
# --------------------------------------------------------------------------------------
class KoppenPredictor:
    """Wraps a model bundle written by koppen_train.py. Call it like a function."""

    def __init__(self, bundle):
        self.booster = bundle["booster"]
        self.classes = np.array(bundle["classes"])
        self.n_classes = len(self.classes)
        self.epoch = bundle.get("epoch", "0ka")
        self.kernel_size = bundle["kernel_size"]
        self.vector_kernel_size = bundle.get("vector_kernel_size", self.kernel_size)
        self.include_water_vectors = bundle.get("include_water_vectors", True)
        self.water_power = bundle.get("water_power", 2)
        self.scale_ew = bundle.get("scale_ew_cos_lat", False)
        self.include_latitude = bundle["includes_latitude"]
        self.requires_anchor = bundle["use_anchor"]
        self.feature_dim = bundle["feature_dim"]
        self.best_iteration = bundle.get("best_iteration")
        self._KK = self.kernel_size**2
        self._static_key = None
        self._static = None

    @classmethod
    def load(cls, path):
        return cls(joblib.load(path))

    # ---- feature building -------------------------------------------------------------
    def _static_features(self, map_grid, progress=True):
        key = hashlib.sha1(map_grid.tobytes()).hexdigest()
        if key != self._static_key:  # cached: reuse when the same map is passed again
            self._static = build_static_spatial_features(
                map_grid,
                kernel_size=self.kernel_size,
                vector_kernel_size=self.vector_kernel_size,
                include_water_vectors=self.include_water_vectors,
                water_power=self.water_power,
                scale_ew=self.scale_ew,
                include_latitude=self.include_latitude,
                progress=progress,
            )
            self._static_key = key
        return self._static

    def _encode(self, koppen):
        arr = np.asarray(koppen)
        if arr.shape != (N_LAT, N_LON):
            raise ValueError(f"anchor_koppen must have shape ({N_LAT}, {N_LON}); got {arr.shape}")
        arr = arr.astype("<U4")
        order = np.argsort(self.classes)
        pos = np.searchsorted(self.classes[order], arr)
        pos = np.clip(pos, 0, self.n_classes - 1)
        code = order[pos]
        ok = (self.classes[code] == arr) & (arr != "")
        return np.where(ok, code, -1).astype(np.int16).reshape(N_PIXELS)

    def _predict_proba(self, X, chunk=200_000):
        out = np.empty((len(X), self.n_classes), dtype=np.float32)
        ni = self.best_iteration if self.best_iteration else None
        for i in range(0, len(X), chunk):
            out[i : i + chunk] = self.booster.predict(X[i : i + chunk], num_iteration=ni)
        return out

    # ---- public call ------------------------------------------------------------------
    def __call__(
        self,
        map_vars,
        temperature,
        *,
        mask=None,
        epoch=None,
        anchor_koppen=None,
        anchor_temperature=None,
        years_since_anchor=None,
        return_proba=False,
        show_progress=True,
    ):
        """Predict Köppen classes.

        Returns a (180, 360) array of strings ("" where nothing is predicted) for a 2-D
        temperature input, or (n, 180, 360) for a stack. With return_proba=True returns
        (labels, probabilities) where probabilities has a trailing class axis (self.classes).

        mask: optional bool (180, 360). Default = pixels with land_area_fraction > 0.
        """
        map_grid = _to_map_grid(map_vars, epoch or self.epoch)
        temps, single = _to_temp_grids(temperature)
        n_steps = len(temps)
        static = self._static_features(map_grid, progress=show_progress)
        S = static.shape[1]
        if S + self._KK + (3 if self.requires_anchor else 0) != self.feature_dim:
            raise ValueError(
                f"feature layout mismatch: this model expects {self.feature_dim} features but the "
                f"saved settings produce {S + self._KK + (3 if self.requires_anchor else 0)}. "
                "Was the model trained with different kernel/vector settings?"
            )

        pix_mask = (map_grid[..., 2] > 0) if mask is None else np.asarray(mask, dtype=bool)
        if pix_mask.shape != (N_LAT, N_LON):
            raise ValueError(f"mask must have shape ({N_LAT}, {N_LON})")
        pix_mask = pix_mask.reshape(N_PIXELS)

        if self.requires_anchor:
            if anchor_koppen is None or anchor_temperature is None or years_since_anchor is None:
                raise ValueError(
                    "this model was trained with the anchor feature: pass anchor_koppen, "
                    "anchor_temperature and years_since_anchor (or retrain with --no-anchor)."
                )
            a_code = self._encode(anchor_koppen)
            a_T = np.asarray(anchor_temperature, dtype=np.float32).reshape(N_PIXELS)
            lags = np.broadcast_to(np.asarray(years_since_anchor, dtype=np.float32), (n_steps,))
            pix_mask = pix_mask & (a_code >= 0)

        idx = np.flatnonzero(pix_mask)
        labels = np.full((n_steps, N_PIXELS), "", dtype="<U4")
        proba = np.full((n_steps, N_PIXELS, self.n_classes), np.nan, dtype=np.float32) if return_proba else None

        bar = StepBar(n_steps, "kernelizing temperature + predicting", unit="step") if show_progress else None
        for t in range(n_steps):
            if len(idx) == 0:
                break
            t_grid = temps[t].reshape(N_LAT, N_LON, 1)
            t_kernel = build_neighbourhood_features_for_every_pixel(t_grid, self.kernel_size).reshape(N_PIXELS, -1)
            X = np.empty((len(idx), self.feature_dim), dtype=np.float32)
            X[:, :S] = static[idx]
            X[:, S : S + self._KK] = t_kernel[idx]
            if self.requires_anchor:
                X[:, S + self._KK] = a_code[idx]
                X[:, S + self._KK + 1] = lags[t]
                X[:, S + self._KK + 2] = temps[t].reshape(N_PIXELS)[idx] - a_T[idx]
            p = self._predict_proba(X)
            labels[t, idx] = self.classes[p.argmax(axis=1)]
            if return_proba:
                proba[t, idx] = p
            if bar:
                bar.update()
        if bar:
            bar.close()

        labels = labels.reshape(n_steps, N_LAT, N_LON)
        if single:
            labels = labels[0]
        if not return_proba:
            return labels
        proba = proba.reshape(n_steps, N_LAT, N_LON, self.n_classes)
        return labels, (proba[0] if single else proba)


_CACHE = {}


def load_model(path):
    """Load a koppen_*.joblib bundle and return a callable predictor."""
    return KoppenPredictor.load(path)


def predict_koppen(model_path, map_vars, temperature, **kwargs):
    """One-shot convenience: loads (and caches) the model, then predicts."""
    if model_path not in _CACHE:
        _CACHE[model_path] = load_model(model_path)
    return _CACHE[model_path](map_vars, temperature, **kwargs)


# --------------------------------------------------------------------------------------
# Command line: predict one year from CSVs and write a CSV
# --------------------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(description="Predict a Köppen map from MapVariables.csv + a temperature CSV.")
    ap.add_argument("--model", required=True, help="koppen_*.joblib written by koppen_train.py")
    ap.add_argument("--map-csv", required=True, help="MapVariables.csv")
    ap.add_argument("--temp-csv", required=True, help="wide temperature CSV (lat, lon, <year> columns)")
    ap.add_argument("--year", required=True, help="year column to use")
    ap.add_argument("--scenario", default=None, help="scenario name, if the temperature CSV has a scenario column")
    ap.add_argument("--epoch", default=None, choices=["0ka", "21ka"])
    ap.add_argument("--out", default="koppen_prediction.csv")
    a = ap.parse_args()

    predictor = load_model(a.model)
    if predictor.requires_anchor:
        raise SystemExit("This model needs anchor inputs; use the Python API, or retrain with --no-anchor.")
    usecols = (["scenario"] if a.scenario else []) + ["lat", "lon", str(a.year)]
    temp = pd.read_csv(a.temp_csv, usecols=usecols)
    if a.scenario:
        temp = temp[temp["scenario"] == a.scenario].drop(columns="scenario")
    labels = predictor(pd.read_csv(a.map_csv), temp, epoch=a.epoch)
    out = grid_to_dataframe(labels)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out):,} pixels to {a.out}")


if __name__ == "__main__":
    _cli()
