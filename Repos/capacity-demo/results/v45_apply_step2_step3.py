#!/usr/bin/env python3
"""
Framework v4.5 — Apply Step 2 Threshold Gates + Step 3 Selection Gates to REAL pipeline outputs.

Goal
  - Step 2: evaluate threshold gates (fit, gluing, UV, isolation) on produced metrics.
  - Step 3: evaluate selection (exercise gap, danger, committed error) from selection logs.

This script is intentionally robust to different output layouts:
  - It auto-discovers likely metric files (json/jsonl/csv) under a root directory.
  - If a metric is missing, it marks the gate as SKIP with an explanation (never silently PASS).

Usage
  python v45_apply_step2_step3.py --root /path/to/your/run_outputs

Outputs
  - gates_report.json
  - gates_report.md
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Helpers
# ----------------------------

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _find_files(root: Path, patterns: List[str]) -> List[Path]:
    found: List[Path] = []
    for pat in patterns:
        found.extend(root.rglob(pat))
    # de-dupe while preserving order
    seen = set()
    out = []
    for p in found:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(x)
        return float(x)
    except Exception:
        return None

def _norm2(vec: List[float]) -> float:
    return math.sqrt(sum(v*v for v in vec))

def _fro_norm(mat: List[List[float]]) -> float:
    return math.sqrt(sum((v*v) for row in mat for v in row))

def _as_set(x: Any) -> set:
    if x is None:
        return set()
    if isinstance(x, (list, tuple, set)):
        return set(x)
    # allow comma-separated strings
    if isinstance(x, str):
        return set([s.strip() for s in x.split(",") if s.strip()])
    return set([x])

# ----------------------------
# Gate definitions
# ----------------------------

@dataclass
class GateResult:
    name: str
    status: str  # PASS / FAIL / SKIP
    details: Dict[str, Any]

def pass_(name: str, **details) -> GateResult:
    return GateResult(name=name, status="PASS", details=details)

def fail(name: str, **details) -> GateResult:
    return GateResult(name=name, status="FAIL", details=details)

def skip(name: str, **details) -> GateResult:
    return GateResult(name=name, status="SKIP", details=details)

# ----------------------------
# Step 2: threshold gates
# ----------------------------

def gate_fit(metrics: Dict[str, Any]) -> GateResult:
    """
    Canonical: ||G_EFT - G_C|| <= eps_fit(C)
    Acceptable real proxies:
      - fit_error <= eps_fit
      - or, if you log both G_EFT and G_C as matrices/vectors: compute norm and compare
    """
    eps = _safe_float(metrics.get("eps_fit")) or _safe_float(metrics.get("epsilon_fit")) or _safe_float(metrics.get("eps_fit_C"))
    # direct scalar fit error path
    fit_err = _safe_float(metrics.get("fit_error")) or _safe_float(metrics.get("fit_err"))
    if fit_err is not None and eps is not None:
        return pass_("fit", fit_error=fit_err, eps_fit=eps) if fit_err <= eps else fail("fit", fit_error=fit_err, eps_fit=eps)
    # matrix / vector norm path
    G_eft = metrics.get("G_EFT") or metrics.get("G_eft")
    G_c = metrics.get("G_C") or metrics.get("G_c") or metrics.get("G_cap")
    if G_eft is not None and G_c is not None and eps is not None:
        # determine vector vs matrix
        if isinstance(G_eft, list) and len(G_eft) > 0 and isinstance(G_eft[0], list):
            dist = _fro_norm(G_eft) - 0.0  # placeholder; compare difference if both are matrices
            # If both are matrices, compute Frobenius norm of difference
            if isinstance(G_c, list) and len(G_c) > 0 and isinstance(G_c[0], list):
                diff = [[float(G_eft[i][j]) - float(G_c[i][j]) for j in range(len(G_eft[0]))] for i in range(len(G_eft))]
                dist = _fro_norm(diff)
            else:
                return skip("fit", reason="G_EFT is matrix but G_C is not; cannot diff cleanly")
        else:
            # treat as vector
            if isinstance(G_c, list):
                if len(G_eft) != len(G_c):
                    return skip("fit", reason="vector length mismatch", len_G_EFT=len(G_eft), len_G_C=len(G_c))
                diff = [float(G_eft[i]) - float(G_c[i]) for i in range(len(G_eft))]
                dist = _norm2(diff)
            else:
                return skip("fit", reason="G_EFT is vector but G_C is not; cannot diff cleanly")
        return pass_("fit", dist=dist, eps_fit=eps) if dist <= eps else fail("fit", dist=dist, eps_fit=eps)

    return skip("fit", reason="missing fit_error/eps_fit or missing G_EFT/G_C and eps_fit", available_keys=sorted(metrics.keys())[:50])

def gate_gluing(metrics: Dict[str, Any]) -> GateResult:
    """
    Canonical: Delta_ab(ell) <= k * N_geo(C_geo)^(-1/2)
    Proxy: overlap_delta <= k / sqrt(N_geo) or overlap_delta_max similarly.
    """
    delta = _safe_float(metrics.get("overlap_delta")) or _safe_float(metrics.get("overlap_delta_max")) or _safe_float(metrics.get("delta_overlap"))
    k = _safe_float(metrics.get("k_glue")) or _safe_float(metrics.get("k")) or _safe_float(metrics.get("glue_k"))
    N_geo = _safe_float(metrics.get("N_geo")) or _safe_float(metrics.get("n_geo")) or _safe_float(metrics.get("Ngeo"))
    if delta is None or k is None or N_geo is None:
        return skip("gluing", reason="need overlap_delta, k_glue, and N_geo", overlap_delta=delta, k_glue=k, N_geo=N_geo)
    if N_geo <= 0:
        return skip("gluing", reason="N_geo <= 0", N_geo=N_geo)
    thresh = k / math.sqrt(N_geo)
    return pass_("gluing", overlap_delta=delta, threshold=thresh, k_glue=k, N_geo=N_geo) if delta <= thresh else fail("gluing", overlap_delta=delta, threshold=thresh, k_glue=k, N_geo=N_geo)

def gate_uv(metrics: Dict[str, Any]) -> GateResult:
    """
    Canonical: component UV thresholds (e.g., lambda1, lambda_int, etc. below some max, or above some min)
    Since different pipelines define these differently, we support two common patterns:
      (A) lambda_* <= uv_max_*
      (B) lambda_* >= uv_min_*
    """
    checks = []
    for key in ["lambda1", "lambda_1", "λ1", "lam1"]:
        v = _safe_float(metrics.get(key))
        if v is not None:
            checks.append(("lambda1", v))
            break
    for key in ["lambda_int", "λ_int", "lam_int", "lambdaInt"]:
        v = _safe_float(metrics.get(key))
        if v is not None:
            checks.append(("lambda_int", v))
            break
    if not checks:
        return skip("uv", reason="no lambda metrics found", available_keys=sorted(metrics.keys())[:50])

    # build thresholds
    failures = []
    evals = []
    for name, val in checks:
        uv_max = _safe_float(metrics.get(f"uv_max_{name}")) or _safe_float(metrics.get(f"uv_{name}_max")) or _safe_float(metrics.get(f"{name}_uv_max"))
        uv_min = _safe_float(metrics.get(f"uv_min_{name}")) or _safe_float(metrics.get(f"uv_{name}_min")) or _safe_float(metrics.get(f"{name}_uv_min"))
        if uv_max is None and uv_min is None:
            evals.append({"name": name, "value": val, "status": "SKIP", "reason": "no uv_min/uv_max for this lambda"})
            continue
        ok = True
        if uv_max is not None and val > uv_max:
            ok = False
            failures.append({"name": name, "value": val, "uv_max": uv_max})
        if uv_min is not None and val < uv_min:
            ok = False
            failures.append({"name": name, "value": val, "uv_min": uv_min})
        evals.append({"name": name, "value": val, "uv_max": uv_max, "uv_min": uv_min, "status": "PASS" if ok else "FAIL"})

    if all(e["status"] == "SKIP" for e in evals):
        return skip("uv", reason="lambdas present but no uv thresholds provided", lambdas=evals)
    return pass_("uv", lambdas=evals) if not failures else fail("uv", lambdas=evals, failures=failures)

def gate_isolation(metrics: Dict[str, Any]) -> GateResult:
    """
    Canonical: cross-axis isolation (non-geo instability should not contaminate geo beyond tolerance).
    Proxy patterns supported:
      - isolation_metric <= isolation_eps
      - leakage <= leakage_eps
    """
    iso = _safe_float(metrics.get("isolation_metric")) or _safe_float(metrics.get("leakage")) or _safe_float(metrics.get("cross_leakage"))
    eps = _safe_float(metrics.get("isolation_eps")) or _safe_float(metrics.get("leakage_eps")) or _safe_float(metrics.get("eps_isolation"))
    if iso is None or eps is None:
        return skip("isolation", reason="need isolation_metric/leakage and isolation_eps", isolation_metric=iso, isolation_eps=eps)
    return pass_("isolation", isolation_metric=iso, isolation_eps=eps) if iso <= eps else fail("isolation", isolation_metric=iso, isolation_eps=eps)

# ----------------------------
# Step 3: selection gates
# ----------------------------

def selection_eval(records: List[Dict[str, Any]]) -> GateResult:
    """
    Selection record schema (flexible):
      - accessible: list[str] or set-like
      - selected: list[str]
      - committed: list[str] (optional)
      - ptr: dict[item->score] or list of {item, score}
      - truth: dict[item->bool] or list of {item, is_true} (optional; may be external)
      - theta_ptr: float (optional, else read from record or global)
    Outputs:
      - exercise_gap size
      - danger items (selected AND ptr>=theta AND (truth==False if truth provided else unknown))
      - committed_error subset if committed provided
    """
    if not records:
        return skip("selection", reason="no selection records")

    total_gap = 0
    total_access = 0
    total_selected = 0
    danger_count = 0
    committed_error_count = 0
    unknown_truth_danger = 0

    examples = []

    for r in records[:5000]:  # safety cap
        A = _as_set(r.get("accessible") or r.get("A_accessible") or r.get("access_set"))
        S = _as_set(r.get("selected") or r.get("A_selected") or r.get("selected_set"))
        committed = _as_set(r.get("committed") or r.get("A_committed") or r.get("commit_set"))

        gap = A - S
        total_gap += len(gap)
        total_access += len(A)
        total_selected += len(S)

        # pointer stability scores
        ptr = r.get("ptr") or r.get("pointer") or r.get("pointer_scores")
        ptr_scores: Dict[str, float] = {}
        if isinstance(ptr, dict):
            ptr_scores = {str(k): float(v) for k, v in ptr.items() if _safe_float(v) is not None}
        elif isinstance(ptr, list):
            # allow list of dicts
            for it in ptr:
                if isinstance(it, dict) and "item" in it and ("score" in it or "ptr" in it):
                    ptr_scores[str(it["item"])] = float(it.get("score", it.get("ptr")))
        theta = _safe_float(r.get("theta_ptr")) or _safe_float(r.get("ptr_threshold")) or _safe_float(r.get("theta")) or 0.8

        # truth labels (optional)
        truth = r.get("truth") or r.get("labels") or r.get("is_true")
        truth_map: Dict[str, Optional[bool]] = {}
        if isinstance(truth, dict):
            for k, v in truth.items():
                if isinstance(v, bool):
                    truth_map[str(k)] = v
                elif v in (0, 1):
                    truth_map[str(k)] = bool(v)
        elif isinstance(truth, list):
            for it in truth:
                if isinstance(it, dict) and "item" in it and ("is_true" in it or "truth" in it):
                    truth_map[str(it["item"])] = bool(it.get("is_true", it.get("truth")))

        # danger: selected items that are pointer-stable
        stable_selected = {x for x in S if ptr_scores.get(str(x), 0.0) >= theta}
        for item in stable_selected:
            t = truth_map.get(str(item), None) if truth_map else None
            if t is False:
                danger_count += 1
            elif t is None:
                unknown_truth_danger += 1

        # committed error if committed set exists and truth exists
        if committed and truth_map:
            for item in committed:
                if truth_map.get(str(item)) is False:
                    committed_error_count += 1

        if len(examples) < 5:
            examples.append({
                "A_accessible": sorted(list(A))[:20],
                "A_selected": sorted(list(S))[:20],
                "gap_size": len(gap),
                "stable_selected": sorted(list(stable_selected))[:20],
                "theta_ptr": theta
            })

    # Basic sanity: selection exists if selected not always empty
    if total_access == 0:
        return skip("selection", reason="no accessible items in records")
    avg_gap = total_gap / max(1, len(records))
    gap_rate = total_gap / max(1, total_access)
    selected_rate = total_selected / max(1, total_access)

    details = {
        "n_records": len(records),
        "avg_gap_size": avg_gap,
        "gap_rate": gap_rate,
        "selected_rate": selected_rate,
        "danger_count_false_if_truth_present": danger_count,
        "danger_count_truth_unknown": unknown_truth_danger,
        "committed_error_count": committed_error_count,
        "examples": examples,
        "note": "danger_count_false_if_truth_present requires truth labels. If you do not provide labels, danger is reported as 'truth unknown'."
    }

    # Gate interpretation:
    # - We do not declare PASS/FAIL without user-specified tolerances. We mark PASS if no false-danger and no committed errors given labels.
    # - Otherwise FAIL if labeled false-danger or committed errors exist.
    if danger_count == 0 and committed_error_count == 0 and (unknown_truth_danger == 0):
        return pass_("selection", **details)
    # If labels exist and any committed errors exist => FAIL.
    if committed_error_count > 0 or danger_count > 0:
        return fail("selection", **details)
    # Otherwise we can't say; SKIP but with metrics.
    return skip("selection", **details)

# ----------------------------
# Discovery: load metrics + selection logs
# ----------------------------

def load_metrics(root: Path) -> Tuple[Dict[str, Any], List[str]]:
    """
    Combine metrics from the most relevant json files found.
    Priority:
      - run_metrics.json / metrics.json / summary.json
      - then any *_metrics.json
    """
    patterns = ["run_metrics.json", "metrics.json", "summary.json", "*metrics*.json", "*_summary*.json"]
    files = _find_files(root, patterns)
    notes = []
    if not files:
        return {}, ["No metrics json found."]
    # pick the smallest set: prefer exact matches
    preferred = [p for p in files if p.name in ("run_metrics.json", "metrics.json", "summary.json")]
    use = preferred[:3] if preferred else files[:3]
    merged: Dict[str, Any] = {}
    notes.append(f"Using metrics files: {[str(p) for p in use]}")
    for p in use:
        try:
            data = _read_json(p)
            if isinstance(data, dict):
                merged.update(data)
            else:
                merged[p.stem] = data
        except Exception as e:
            notes.append(f"Failed to read {p}: {e}")
    return merged, notes

def load_selection(root: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    patterns = ["*selection*.jsonl", "*select*.jsonl", "*selection*.json", "*select*.json"]
    files = _find_files(root, patterns)
    notes = []
    if not files:
        return [], ["No selection logs found."]
    # choose first jsonl if available
    jsonl = [p for p in files if p.suffix.lower() == ".jsonl"]
    if jsonl:
        p = jsonl[0]
        notes.append(f"Using selection jsonl: {p}")
        try:
            recs = list(_iter_jsonl(p))
            return recs, notes
        except Exception as e:
            notes.append(f"Failed to read {p}: {e}")
    # else json list
    p = files[0]
    notes.append(f"Using selection json: {p}")
    try:
        data = _read_json(p)
        if isinstance(data, list):
            return data, notes
        if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
            return data["records"], notes
        return [], notes + ["Selection file did not contain a list of records."]
    except Exception as e:
        return [], notes + [f"Failed to read {p}: {e}"]

# ----------------------------
# Reporting
# ----------------------------

def to_md(gates: List[GateResult], discovery_notes: List[str], selection_notes: List[str]) -> str:
    lines = []
    lines.append("Framework v4.5 — Step 2 + Step 3 Gates Report")
    lines.append("")
    lines.append("Discovery")
    for n in discovery_notes:
        lines.append(f"- {n}")
    for n in selection_notes:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("Results")
    for g in gates:
        lines.append(f"- {g.name}: {g.status}")
        # keep details short
        if g.details:
            # include a few key fields
            keys = list(g.details.keys())
            show = {}
            for k in keys:
                if k in ("available_keys", "examples"):
                    continue
                show[k] = g.details[k]
                if len(show) >= 6:
                    break
            if show:
                lines.append(f"  - details: {json.dumps(show, ensure_ascii=False)}")
            if "examples" in g.details:
                lines.append("  - examples included in JSON report")
    lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Root directory containing a run's outputs")
    ap.add_argument("--out", type=str, default="gates_report", help="Output basename (without extension)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    discovery_notes = [f"Root: {root}"]
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    metrics, mnotes = load_metrics(root)
    discovery_notes.extend(mnotes)

    selection_records, snotes = load_selection(root)

    gates: List[GateResult] = []
    gates.append(gate_fit(metrics))
    gates.append(gate_gluing(metrics))
    gates.append(gate_uv(metrics))
    gates.append(gate_isolation(metrics))
    gates.append(selection_eval(selection_records))

    out_json = root / f"{args.out}.json"
    out_md = root / f"{args.out}.md"

    payload = {
        "root": str(root),
        "metrics_keys": sorted(metrics.keys()),
        "discovery_notes": discovery_notes,
        "selection_notes": snotes,
        "gates": [asdict(g) for g in gates],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(to_md(gates, discovery_notes, snotes), encoding="utf-8")

    # console summary
    print(out_md.read_text(encoding="utf-8"))
    # exit code: fail if any FAIL
    if any(g.status == "FAIL" for g in gates):
        raise SystemExit(2)

if __name__ == "__main__":
    main()
