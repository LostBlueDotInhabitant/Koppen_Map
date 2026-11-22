"""Connects the "Draw a map" website to the ridge temperature models and writes a .npy of local temperature.

Put this file in t_reg_predict/ (next to kernelize_exp.py, data_cache/, and the two .joblib models).

A) Live, with the website:
       python drawn_map_server.py                      # serves ../Website on http://localhost:8000
       python drawn_map_server.py --site path/to/Website --port 8000
   Open http://localhost:8000, go to "Draw a map", paint, set CO2 / global temperature, click "Run model on my map".
   Every prediction is also saved as  drawn_maps/predictions_drawn_map.npy  (180, 360) float32, row 0 = south,
   column 0 = lon -179.5  (same convention as predict_ridge_dual3.py).

B) Offline, no server: use the drawn_map.json that the page's "Save drawn map" button downloads
       python drawn_map_server.py --json drawn_map.json --co2 420 --temp 14.5 --out my_world.npy

The website page is NOT modified. It already talks to /health, /drawn_map and /predict.
"""
import argparse
import json
import os
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import joblib
import numpy as np

import kernelize_exp as kz

# ----------------------------- settings (edit these) -----------------------------
DATA_DIRECTORY = "data_cache"
OCEAN_MODEL_FILE = "ridge_temperature_model_sea.joblib"
LAND_MODEL_FILE = "ridge_temperature_model.joblib"
METADATA_FILE = "ridge_temperature_model_meta.json"     # kernel settings used when the models were trained
KOPPEN_TREE_FILE = None                                 # optional joblib of a Koppen tree; None = no Koppen
SAVE_DIRECTORY = "drawn_maps"
NPY_NAME = "predictions_drawn_map.npy"
ALLOWED_ORIGINS = ("https://lostbluedotinhabitant.github.io",)
CO2_SLIDER = (100, 1300, 420)            # (min, max, start) shown on the page
TEMPERATURE_SLIDER = (4, 22, 14.5)

# keys the page sends, in kernelize's channel order
PAGE_KEYS = ["altitude", "ice_thickness", "land_area_fraction", "ice_area_fraction"]


# ======================= model wrapper =======================
class TemperaturePredictor:
    """Loads the land + ocean ridge models once; predicts a (180, 360) temperature grid for any map."""

    def __init__(self, tree=None):
        with open(METADATA_FILE) as f:
            self.meta = json.load(f)
        kz.configure_data_sources(github_base_url=None, temperature_directory=DATA_DIRECTORY)
        kz.configure_kernel(kernel_size=self.meta["kernel_size"],
                            water_vector_kernel_size=self.meta["water_vector_kernel_size"],
                            water_inverse_distance_power=self.meta["water_inverse_distance_power"])
        self.ocean_model = joblib.load(OCEAN_MODEL_FILE)
        self.land_model = joblib.load(LAND_MODEL_FILE)
        self.tree = tree

    def predict(self, map_variables, global_mean_temperature, co2_ppm):
        features = kz.make_temperature_regression_features(
            self.meta["kernel_size"], global_mean_temperature, co2_ppm, map_variables=map_variables)
        if features.shape[1] != self.meta["n_features"]:
            raise RuntimeError(f"Feature count {features.shape[1]} != model's {self.meta['n_features']}; "
                               "kernelize settings do not match how the models were trained.")
        # same land/ocean split as predict_ridge_dual3.py: land_fraction > 0 -> land model
        is_land = map_variables[..., 2].ravel() > 0
        flat = np.empty(features.shape[0], dtype=np.float64)
        if is_land.any():
            flat[is_land] = self.land_model.predict(features[is_land])
        if (~is_land).any():
            flat[~is_land] = self.ocean_model.predict(features[~is_land])
        return kz.reshape_predictions_to_grid(flat)

    def koppen(self, map_variables, temperature_grid):
        """(180, 360) array of labels. The page requires this field; without a tree every pixel is 'n/a'."""
        if self.tree is None:
            return np.full((kz.NUM_LAT_ROWS, kz.NUM_LON_COLS), "n/a", dtype=object)
        features = kz.make_koppen_tree_features(self.meta["kernel_size"], temperature_grid,
                                                map_variables=map_variables)
        return kz.reshape_predictions_to_grid(self.tree.predict(features))


# ======================= conversion & saving =======================
def map_variables_from_page_payload(payload):
    """4 flat lists of 64,800 numbers (row 0 = lat -89.5, col 0 = lon -179.5) -> (180, 360, 4) float32."""
    channels = []
    for key in PAGE_KEYS:
        values = np.asarray(payload[key], dtype=np.float32)
        if values.size != kz.NUM_PIXELS:
            raise ValueError(f"'{key}' has {values.size} values, expected {kz.NUM_PIXELS}")
        if not np.isfinite(values).all():
            raise ValueError(f"'{key}' contains NaN/inf")
        channels.append(values.reshape(kz.NUM_LAT_ROWS, kz.NUM_LON_COLS))
    return np.stack(channels, axis=-1)


def save_outputs(map_variables, temperature_grid, global_mean_temperature, co2_ppm, npy_path=None):
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    np.savez(os.path.join(SAVE_DIRECTORY, "latest.npz"), map_variables=map_variables,
             channel_names=np.array(PAGE_KEYS), global_mean_temperature_c=global_mean_temperature, co2_ppm=co2_ppm)
    npy_path = npy_path or os.path.join(SAVE_DIRECTORY, NPY_NAME)
    np.save(npy_path, temperature_grid.astype(np.float32))
    return npy_path


# ======================= the server =======================
def run_server(predictor, site_folder="../Website", port=8000):
    state = {"map_variables": None}
    lock = threading.Lock()

    # tolerate files that were downloaded twice by a browser ("index(1).html", "mapvariables(1).json")
    def resolve(name):
        if os.path.exists(os.path.join(site_folder, name)):
            return name
        stem, ext = os.path.splitext(name)
        for candidate in sorted(os.listdir(site_folder)):
            if candidate.startswith(stem + "(") and candidate.endswith(ext):
                return candidate
        return name

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=site_folder, **kwargs)

        def translate_path(self, path):
            clean = path.split("?")[0]
            if clean in ("/", "/index.html"):
                path = "/" + resolve("index.html")
            elif clean == "/mapvariables.json":
                path = "/" + resolve("mapvariables.json")
            return super().translate_path(path)

        def _cors(self):
            origin = self.headers.get("Origin")
            if origin and origin in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Vary", "Origin")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def log_message(self, *args):
            pass

        def _send_json(self, obj, status=200):
            body = json.dumps(obj, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/health":
                return self._send_json({"kernelize": True, "predictions": True,
                                        "sliders": {"co2": list(CO2_SLIDER), "temp": list(TEMPERATURE_SLIDER)}})
            return super().do_GET()

        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                route = self.path.split("?")[0]
                with lock:
                    if route == "/drawn_map":
                        state["map_variables"] = map_variables_from_page_payload(payload)
                        os.makedirs(SAVE_DIRECTORY, exist_ok=True)
                        np.savez(os.path.join(SAVE_DIRECTORY, "latest.npz"),
                                 map_variables=state["map_variables"], channel_names=np.array(PAGE_KEYS))
                        return self._send_json({"ok": True})
                    if route == "/predict":
                        if state["map_variables"] is None:
                            return self._send_json({"error": "no drawn map received yet"}, 409)
                        temperature = float(payload["global_mean_temperature"])
                        co2 = float(payload["co2_ppm"])
                        grid_t = predictor.predict(state["map_variables"], temperature, co2)
                        grid_k = predictor.koppen(state["map_variables"], grid_t)
                        path = save_outputs(state["map_variables"], grid_t, temperature, co2)
                        print(f"predicted (T_global={temperature}, CO2={co2}) -> {path}  "
                              f"[min {grid_t.min():.1f}, mean {grid_t.mean():.1f}, max {grid_t.max():.1f} C]")
                        return self._send_json({"temperature": np.round(grid_t, 2).ravel().tolist(),
                                                "koppen": [str(label) for label in grid_k.ravel()]})
                self._send_json({"error": "unknown route"}, 404)
            except Exception as error:
                traceback.print_exc()
                self._send_json({"error": str(error)}, 400)

    if not os.path.isdir(site_folder):
        raise SystemExit(f"Website folder not found: {site_folder}  (use --site)")
    server = ThreadingHTTPServer(("localhost", port), Handler)
    print(f"Open http://localhost:{port}  ->  'Draw a map' tab. Predictions are saved to "
          f"{os.path.join(SAVE_DIRECTORY, NPY_NAME)}")
    server.serve_forever()


# ======================= command line =======================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default="../Website", help="folder with index.html + mapvariables.json")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--json", help="offline mode: a drawn_map.json downloaded from the page")
    parser.add_argument("--co2", type=float, default=CO2_SLIDER[2], help="offline mode: CO2 ppm")
    parser.add_argument("--temp", type=float, default=TEMPERATURE_SLIDER[2], help="offline mode: global mean temp C")
    parser.add_argument("--out", default="predictions_drawn_map.npy", help="offline mode: output .npy path")
    args = parser.parse_args()

    tree = joblib.load(KOPPEN_TREE_FILE) if KOPPEN_TREE_FILE else None
    predictor = TemperaturePredictor(tree)

    if args.json:
        with open(args.json) as f:
            map_variables = map_variables_from_page_payload(json.load(f))
        grid = predictor.predict(map_variables, args.temp, args.co2)
        np.save(args.out, grid.astype(np.float32))
        print(f"Saved {args.out} {grid.shape}  min {grid.min():.1f}  mean {grid.mean():.1f}  max {grid.max():.1f} C")
    else:
        run_server(predictor, args.site, args.port)


if __name__ == "__main__":
    main()
