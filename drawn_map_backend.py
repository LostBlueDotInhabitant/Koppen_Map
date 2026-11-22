"""Backend for the website's 'Draw a map' tab. It IMPORTS kernelize.py and never modifies it.

Run it (same kernelize settings you trained with):
    import kernelize as kz, drawn_map_backend as bk
    kz.configure_data_sources(...); kz.configure_kernel(...)
    ... fit `regressor` and `tree` ...
    bk.run_server(regressor, tree, site_folder="website")     # then open http://localhost:8000
    bk.run_server(site_folder="website")                      # no models yet: drawing only, maps still saved

What happens automatically while you draw:
    page --(POST /drawn_map)--> map converted to kernelize's (180, 360, 4) array and saved to drawn_maps/latest.npz
    page --(POST /predict)----> kernelize features -> regressor -> temperature -> tree -> Koppen -> back to the page

Later, from Python, with no website:
    mv, settings = bk.load_saved_drawn_map("drawn_maps/latest.npz")
    temperature, koppen = bk.predict_from_map_variables(mv, regressor, tree,
                                                        settings["global_mean_temperature_c"], settings["co2_ppm"])
"""
import json, os, threading, traceback
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np
import kernelize as kz

# keys the page sends, in kernelize's channel order: altitude, ice thickness, land fraction, ice fraction
PAGE_KEYS = ["altitude", "ice_thickness", "land_area_fraction", "ice_area_fraction"]


# ======================= conversion, saving, prediction =======================
def map_variables_from_page_payload(payload):
    """JSON from the page (4 flat lists of 64,800 numbers, row 0 = lat -89.5, column 0 = lon -179.5)
    -> (180, 360, 4) float32 array, the same layout kernelize's feature builders take as map_variables=."""
    channels = []
    for key in PAGE_KEYS:
        values = np.asarray(payload[key], dtype=np.float32)
        if values.size != kz.NUM_PIXELS:
            raise ValueError(f"'{key}' has {values.size} values, expected {kz.NUM_PIXELS}")
        channels.append(values.reshape(kz.NUM_LAT_ROWS, kz.NUM_LON_COLS))
    return np.stack(channels, axis=-1)

def save_drawn_map(map_variables, path, settings=None):
    """Save the drawn map (and the CO2 / global-temperature settings, if known) as a .npz file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, map_variables=map_variables, channel_names=np.array(PAGE_KEYS), **(settings or {}))

def load_saved_drawn_map(path):
    """-> (map_variables (180, 360, 4), settings dict with 'co2_ppm' / 'global_mean_temperature_c' if saved)."""
    with np.load(path) as f:
        settings = {k: float(f[k]) for k in f.files if k not in ("map_variables", "channel_names")}
        return f["map_variables"], settings

def predict_from_map_variables(map_variables, regressor, tree, global_mean_temperature, co2_ppm, kernel_size=None):
    """Temperature, then Koppen, for any (180, 360, 4) map. Returns (temperature_grid, koppen_grid), both (180, 360).
    Uses only kernelize's public feature builders, so it follows whatever kernelize settings are configured."""
    temperature_features = kz.make_temperature_regression_features(kernel_size, global_mean_temperature, co2_ppm,
                                                                   map_variables=map_variables)
    temperature_grid = kz.reshape_predictions_to_grid(regressor.predict(temperature_features))
    tree_features = kz.make_koppen_tree_features(kernel_size, temperature_grid, map_variables=map_variables)
    return temperature_grid, kz.reshape_predictions_to_grid(tree.predict(tree_features))


# ======================= the server =======================
def run_server(regressor=None, tree=None, site_folder=".", port=8000, kernel_size=None, save_directory="drawn_maps",
               co2_slider=(180, 1300, 420), temperature_slider=(10, 22, 14.5), block=True,
               allowed_origins=("https://lostbluedotinhabitant.github.io",)):
    """Serve the website (index.html, mapvariables.json, ... from site_folder) and the two endpoints above.
    regressor / tree may be left out (None): drawing-only mode. Every stroke is still saved to latest.npz,
    but /predict is refused until you restart the server with fitted models.
    *_slider = (min, max, starting value) for the page's sliders. block=False returns the running server."""
    state = {"map_variables": None}
    lock = threading.Lock()                                  # one request at a time touches the state
    latest_path = os.path.join(save_directory, "latest.npz")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=site_folder, **kwargs)

        def _cors(self):                                     # lets the hosted page talk to this local server
            origin = self.headers.get("Origin")
            if origin and origin in allowed_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Vary", "Origin")

        def do_OPTIONS(self):                                # CORS preflight
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def log_message(self, *args):                        # slider drags would flood the console
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
            if self.path.split("?")[0] == "/health":         # lets the page detect that a backend is running
                return self._send_json({"kernelize": True, "predictions": regressor is not None and tree is not None, "sliders": {"co2": list(co2_slider), "temp": list(temperature_slider)}})
            return super().do_GET()

        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                route = self.path.split("?")[0]
                with lock:
                    if route == "/drawn_map":
                        state["map_variables"] = map_variables_from_page_payload(payload)
                        save_drawn_map(state["map_variables"], latest_path)
                        return self._send_json({"ok": True})
                    if route == "/predict":
                        if regressor is None or tree is None:
                            return self._send_json({"error": "drawing-only mode: no models loaded"}, 409)
                        if state["map_variables"] is None:
                            return self._send_json({"error": "no drawn map received yet"}, 409)
                        temperature, co2 = float(payload["global_mean_temperature"]), float(payload["co2_ppm"])
                        grid_t, grid_k = predict_from_map_variables(state["map_variables"], regressor, tree,
                                                                    temperature, co2, kernel_size)
                        save_drawn_map(state["map_variables"], latest_path,
                                       {"global_mean_temperature_c": temperature, "co2_ppm": co2})
                        return self._send_json({"temperature": np.round(grid_t, 2).ravel().tolist(),
                                                "koppen": [str(label) for label in grid_k.ravel()]})
                self._send_json({"error": "unknown route"}, 404)
            except Exception as error:
                traceback.print_exc()
                self._send_json({"error": str(error)}, 400)

    server = ThreadingHTTPServer(("localhost", port), Handler)
    print(f"Open http://localhost:{port}   (drawn maps are saved to {latest_path})")
    if block:
        server.serve_forever()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
