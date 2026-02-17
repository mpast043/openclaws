"""
Flask web server for the Framework v4.5 Capacity Demo.

Serves the interactive web UI and handles computation API requests.
All computation is done via the dimshift library.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from dimshift import SweepConfig, run_capacity_sweep
from dimshift.capacity import capacity_weights
from dimshift.spectral import eigenvalues_1d, log_return_probability, spectral_dimension

app = Flask(__name__, static_folder="static")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Result persistence (web-specific — saves full per-scan JSON for the UI)
# ---------------------------------------------------------------------------

def _save_web_result(data, label=None):
    ts = time.time()
    rid = hashlib.md5(f"{ts}{data['D']}{data['N']}".encode()).hexdigest()[:10]
    entry = {
        "id": rid,
        "timestamp": ts,
        "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
        "label": label or f"D={data['D']} N={data['N']}",
        "D": data["D"],
        "N": data["N"],
        "total_sites": data["total_sites"],
        "thresholds": data["thresholds"],
        "n_scans": len(data["scans"]),
        "data": data,
    }
    fp = RESULTS_DIR / f"run_{rid}.json"
    with open(fp, "w") as f:
        json.dump(entry, f)
    return entry


def _list_web_results():
    entries = []
    for fp in RESULTS_DIR.glob("run_*.json"):
        with open(fp) as f:
            e = json.load(f)
        entries.append({
            "id": e["id"],
            "timestamp": e.get("timestamp", 0),
            "timestamp_human": e["timestamp_human"],
            "label": e["label"],
            "D": e["D"],
            "N": e["N"],
            "thresholds": e["thresholds"],
            "n_scans": e["n_scans"],
        })
    entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    return entries


def _load_web_result(result_id):
    fp = RESULTS_DIR / f"run_{result_id}.json"
    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    return None


def _delete_web_result(result_id):
    fp = RESULTS_DIR / f"run_{result_id}.json"
    if fp.exists():
        fp.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/compute", methods=["POST"])
def api_compute():
    """Run a capacity scan computation using the dimshift library."""
    params = request.json or {}
    D = int(params.get("D", 3))
    N = int(params.get("N", 64))
    n_steps = int(params.get("n_steps", 30))
    C_min = float(params.get("C_min", 0.05))
    C_max = float(params.get("C_max", 1.0))
    sigma_min = float(params.get("sigma_min", 0.1))
    sigma_max = float(params.get("sigma_max", 200.0))
    n_sigma = int(params.get("n_sigma", 400))
    label = params.get("label", None)

    # Safety limits
    if D > 4:
        return jsonify({"error": "D > 4 not supported (memory)"}), 400
    if N > 64 and D >= 3:
        return jsonify({"error": "N > 64 for D>=3 not supported (memory)"}), 400
    if N > 256:
        return jsonify({"error": "N > 256 not supported"}), 400

    # Run sweep via library
    config = SweepConfig(
        D=D, N=N,
        C_geo_min=C_min, C_geo_max=C_max, C_geo_steps=n_steps,
        sigma_min=sigma_min, sigma_max=sigma_max, n_sigma=n_sigma,
    )
    result = run_capacity_sweep(config)

    # Build per-scan data for the web UI
    scans = []
    for i in range(len(result.C_geo_values)):
        weights = result.weights_list[i]
        scans.append({
            "C_geo": float(result.C_geo_values[i]),
            "weights": weights.tolist(),
            "d_eff_nominal": float(result.d_eff_nominal[i]),
            "n_active_dims": int(np.sum(weights > 0.01)),
            "fraction_active": float(result.d_eff_nominal[i] / D),
            "ds_values": result.ds_matrix[i].tolist(),
            "P_values": np.exp(result.ln_P_matrix[i]).tolist(),
            "ds_plateau": float(result.ds_plateau[i]),
        })

    data = {
        "D": D,
        "N": N,
        "total_sites": N**D,
        "sigma_values": result.sigma_values.tolist(),
        "scans": scans,
        "thresholds": result.thresholds,
    }

    saved = _save_web_result(data, label=label)

    return jsonify({
        "id": saved["id"],
        "D": D,
        "N": N,
        "total_sites": N**D,
        "thresholds": result.thresholds,
        "scans": scans,
        "sigma_values": result.sigma_values.tolist(),
    })


@app.route("/api/results", methods=["GET"])
def api_list_results():
    return jsonify(_list_web_results())


@app.route("/api/results/<result_id>", methods=["GET"])
def api_get_result(result_id):
    result = _load_web_result(result_id)
    if result is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(result)


@app.route("/api/results/<result_id>", methods=["DELETE"])
def api_delete_result(result_id):
    if _delete_web_result(result_id):
        return jsonify({"ok": True})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/reference")
def api_reference():
    return jsonify({
        "references": [
            {"name": "CDT (Causal Dynamical Triangulations)", "ds_UV": 2.0, "ds_IR": 4.0, "D_lattice": 4, "source": "Ambjorn, Jurkiewicz, Loll (2005)"},
            {"name": "Horava-Lifshitz gravity", "ds_UV": 2.0, "ds_IR": 4.0, "D_lattice": 4, "source": "Horava (2009)"},
            {"name": "Asymptotic Safety (FRGE)", "ds_UV": 2.0, "ds_IR": 4.0, "D_lattice": 4, "source": "Lauscher, Reuter (2005)"},
            {"name": "Random comb / branched polymer", "ds_UV": "4/3", "ds_IR": 2.0, "D_lattice": 2, "source": "Durhuus, Jonsson (2006)"},
            {"name": "3D cubic lattice (exact)", "ds_UV": 3.0, "ds_IR": 3.0, "D_lattice": 3, "source": "Standard result"},
            {"name": "2D square lattice (exact)", "ds_UV": 2.0, "ds_IR": 2.0, "D_lattice": 2, "source": "Standard result"},
        ],
        "notes": [
            "d_s = 2 at UV scales is a universal prediction across multiple QG approaches.",
            "Framework v4.5: capacity filtering changes effective dimension of the same substrate.",
            "At full capacity (C_geo=1), d_s equals the lattice dimension D.",
            "At low capacity, fewer spatial directions are resolved → lower effective dimension.",
        ],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Framework v4.5 Capacity Demo")
    print(f"  Running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
