"""Verification engine comparing SelectionResult to TruthLabel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..truth.schema import TruthLabel
from ..truth.schema import load_truth_label
from ..selection.schema import SelectionResult
from .ledger import VerificationLedger, LedgerEntry


class VerificationEngine:
    def __init__(self, high_conf_threshold: float = 0.9, low_conf_threshold: float = 0.5):
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold

    def verify(self, truth_labels: Dict[str, TruthLabel], selections: Dict[str, SelectionResult]) -> VerificationLedger:
        ledger = VerificationLedger()

        for claim_id, truth in truth_labels.items():
            if claim_id not in selections:
                ledger.add(LedgerEntry(
                    claim_id=claim_id,
                    status="MISSING",
                    match=False,
                    predicted=None,
                    truth=self._truth_view(truth),
                    confidence=None,
                    danger_gap=False,
                    witness=None,
                    notes="Selection missing for claim",
                ))
                continue

            sel = selections[claim_id]

            match, notes = self.compare(sel.predicted, truth)

            if match:
                status = "VERIFIED"
            else:
                status = "INCONCLUSIVE" if sel.confidence < self.low_conf_threshold else "FALSIFIED"

            danger = self.check_danger_gap(sel.predicted, truth)

            ledger.add(LedgerEntry(
                claim_id=claim_id,
                status=status,
                match=match,
                predicted=sel.predicted,
                truth=self._truth_view(truth),
                confidence=sel.confidence,
                danger_gap=danger,
                witness=sel.witness_path,
                notes=notes,
            ))

        return ledger

    def compare(self, predicted: Dict[str, Any], truth: TruthLabel) -> Tuple[bool, str]:
        """Return (match, notes)."""

        if truth.truth_type == "boolean":
            t = bool(truth.expected.get("should_pass"))
            # Accept common predicted encodings
            if "should_pass" in predicted:
                p = bool(predicted["should_pass"])
            elif "verdict" in predicted:
                p = str(predicted["verdict"]).upper() in {"PASS", "VERIFIED", "TRUE", "ACCEPT"}
            elif "converges" in predicted:
                p = bool(predicted["converges"])
            else:
                return False, "No comparable boolean key in predicted"

            return (p == t), ("" if p == t else f"Boolean mismatch: predicted={p} truth={t}")

        if truth.truth_type in {"scalar", "dict"}:
            expected = truth.expected
            tolerance = truth.tolerance

            if not isinstance(expected, dict) or not isinstance(tolerance, dict):
                return False, "Truth expected/tolerance must be dict for scalar/dict truth types"

            for k, tol in tolerance.items():
                if k not in predicted or k not in expected:
                    return False, f"Missing key {k} in predicted or truth"

                pv = predicted[k]
                tv = expected[k]

                try:
                    pvf = float(pv)
                    tvf = float(tv)
                except Exception:
                    return False, f"Non-numeric value for key {k}"

                if isinstance(tol, dict):
                    abs_tol = float(tol.get("abs", 0.0))
                    rel_tol = float(tol.get("rel", 0.0))
                    ok = abs(pvf - tvf) <= max(abs_tol, rel_tol * max(1e-12, abs(tvf)))
                else:
                    ok = abs(pvf - tvf) <= float(tol)

                if not ok:
                    return False, f"Out of tolerance for {k}: predicted={pvf} truth={tvf} tol={tol}"

            return True, ""

        return False, f"Unsupported truth_type: {truth.truth_type}"

    def check_danger_gap(self, predicted: Dict[str, Any], truth: TruthLabel) -> bool:
        """Danger gap = false positive on a boolean pass/fail judgement."""

        # Only applies when truth is boolean.
        if truth.truth_type != "boolean":
            return False

        truth_pass = bool(truth.expected.get("should_pass"))
        # infer predicted pass
        pred_pass = None
        if "should_pass" in predicted:
            pred_pass = bool(predicted["should_pass"])
        elif "verdict" in predicted:
            pred_pass = str(predicted["verdict"]).upper() in {"PASS", "VERIFIED", "TRUE", "ACCEPT"}
        elif "converges" in predicted:
            pred_pass = bool(predicted["converges"])

        if pred_pass is None:
            return False

        return (pred_pass is True) and (truth_pass is False)

    @staticmethod
    def _truth_view(truth: TruthLabel) -> Dict[str, Any]:
        # Keep ledger compact; avoid embedding artifact refs unless needed.
        return {
            "truth_type": truth.truth_type,
            "expected": truth.expected,
            "tolerance": truth.tolerance,
            "source": truth.source,
            "confidence": truth.confidence,
            "substrate_id": truth.substrate_id,
            "seed": truth.seed,
        }


def load_truth_set(root: Path, claim_ids: list[str]) -> Dict[str, TruthLabel]:
    tdir = root / "experiments" / "truth" / "truth_labels"
    out: Dict[str, TruthLabel] = {}
    for cid in claim_ids:
        path = tdir / f"{cid}.json"
        out[cid] = load_truth_label(path)
    return out


def step3_gate_metrics(ledger: VerificationLedger, claim_ids: list[str], high_conf_threshold: float = 0.9) -> Dict[str, Any]:
    summary = ledger.summary()

    # Coverage
    covered = sum(1 for e in ledger.entries if e.status != "MISSING")
    coverage = covered / max(1, len(claim_ids))

    # High confidence rate over present
    present = [e for e in ledger.entries if e.status != "MISSING" and e.confidence is not None]
    high_conf = sum(1 for e in present if float(e.confidence) >= high_conf_threshold)
    high_conf_rate = high_conf / max(1, len(present))

    return {
        "selection_truth": summary["accuracy_over_present"],
        "selection_danger_gap": summary["danger_gaps"],
        "selection_coverage": coverage,
        "selection_confidence": high_conf_rate,
        "counts": summary,
    }


def step3_gate_pass(metrics: Dict[str, Any],
                    truth_threshold: float = 0.95,
                    require_zero_danger: bool = True,
                    require_full_coverage: bool = True,
                    high_conf_threshold: float = 0.90) -> Tuple[bool, Dict[str, str]]:
    """Return (pass, reasons)."""

    reasons: Dict[str, str] = {}

    if metrics["selection_truth"] < truth_threshold:
        reasons["selection_truth"] = f"FAIL {metrics['selection_truth']:.3f} < {truth_threshold:.3f}"

    if require_zero_danger and metrics["selection_danger_gap"] != 0:
        reasons["selection_danger_gap"] = f"FAIL danger_gaps={metrics['selection_danger_gap']}"

    if require_full_coverage and metrics["selection_coverage"] < 1.0:
        reasons["selection_coverage"] = f"FAIL coverage={metrics['selection_coverage']:.3f}"

    if metrics["selection_confidence"] < high_conf_threshold:
        reasons["selection_confidence"] = f"FAIL high_conf_rate={metrics['selection_confidence']:.3f}"

    return (len(reasons) == 0), reasons
