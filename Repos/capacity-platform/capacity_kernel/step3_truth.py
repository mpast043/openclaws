"""Step 3 Truth Infrastructure - Ground truth and falsification gates.

Step 3 gates validate that Step 2 predictions actually hold in reality.
They require ground truth labels (outcomes) to falsify incorrect predictions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from collections import deque
import threading
import time
import uuid

import numpy as np


class PredictionOutcome(Enum):
    """Outcomes for a capacity prediction."""
    PENDING = "pending"
    VALIDATED = "validated"      # Prediction held true
    FALSIFIED = "falsified"      # Prediction failed (false positive)
    MISSED = "missed"            # Should have admitted, didn't (false negative)
    TIMEOUT = "timeout"          # Outcome observation window expired


@dataclass
class PredictionRecord:
    """A single capacity prediction awaiting validation."""
    prediction_id: str
    timestamp: datetime
    substrate_id: str
    
    # The capacity vector that was granted
    c_geo: float
    c_int: float
    
    # Gate states at admission (no defaults - required)
    fit_error: float
    gluing_delta: float
    lambda1: float
    lambda_int: float
    isolation: float
    
    # Optional capacity components (with defaults)
    c_gauge: float = 0.0
    c_ptr: float = 0.0
    c_obs: float = 0.0
    
    # Validation window
    outcome_window_seconds: float = 60.0
    
    # Filled by outcome tracking
    outcome: PredictionOutcome = field(default=PredictionOutcome.PENDING)
    outcome_timestamp: Optional[datetime] = None
    outcome_metrics: Dict[str, float] = field(default_factory=dict)
    falsification_reason: Optional[str] = None


@dataclass
class GroundTruthSample:
    """A ground truth observation at a point in time."""
    sample_id: str
    timestamp: datetime
    substrate_id: str
    
    # Actual resource state (ground truth)
    actual_load: float  # 0-1 actual utilization
    actual_stability: float  # 0-1 stability score
    actual_latency_ms: float  # Measured latency
    actual_error_rate: float  # Measured errors
    
    # Corresponding prediction (if any active)
    matched_prediction_id: Optional[str] = None
    
    # Labels for supervised training
    labels: Dict[str, Any] = field(default_factory=dict)


class TruthLedger:
    """
    Records capacity predictions and their outcomes.
    
    This is the ground truth database for Step 3 falsification.
    It tracks what was predicted vs what actually happened.
    """
    
    def __init__(self, max_history: int = 10000):
        self.predictions: Dict[str, PredictionRecord] = {}
        self.ground_truth: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
        
        # Statistics
        self.validation_stats = {
            'total_predictions': 0,
            'validated': 0,
            'falsified': 0,
            'missed': 0,
            'timeouts': 0,
        }
    
    def record_prediction(
        self,
        substrate_id: str,
        c_geo: float,
        c_int: float,
        gate_metrics: Dict[str, float],
        c_gauge: float = 0.0,
        c_ptr: float = 0.0,
        c_obs: float = 0.0,
        outcome_window_seconds: float = 60.0
    ) -> str:
        """
        Record a new capacity prediction for later validation.
        Called when admission is granted.
        """
        pred_id = str(uuid.uuid4())[:12]
        
        record = PredictionRecord(
            prediction_id=pred_id,
            timestamp=datetime.now(),
            substrate_id=substrate_id,
            c_geo=c_geo,
            c_int=c_int,
            c_gauge=c_gauge,
            c_ptr=c_ptr,
            c_obs=c_obs,
            fit_error=gate_metrics.get('fit_error', 0.0),
            gluing_delta=gate_metrics.get('gluing_delta', 0.0),
            lambda1=gate_metrics.get('lambda1', 0.0),
            lambda_int=gate_metrics.get('lambda_int', 0.0),
            isolation=gate_metrics.get('isolation_metric', 0.0),
            outcome_window_seconds=outcome_window_seconds
        )
        
        with self._lock:
            self.predictions[pred_id] = record
            self.validation_stats['total_predictions'] += 1
        
        return pred_id
    
    def record_ground_truth(
        self,
        substrate_id: str,
        actual_load: float,
        actual_stability: float,
        actual_latency_ms: float,
        actual_error_rate: float,
        labels: Optional[Dict] = None
    ) -> str:
        """
        Record an observation of actual substrate state.
        This is the "truth" that validates predictions.
        """
        sample = GroundTruthSample(
            sample_id=str(uuid.uuid4())[:12],
            timestamp=datetime.now(),
            substrate_id=substrate_id,
            actual_load=actual_load,
            actual_stability=actual_stability,
            actual_latency_ms=actual_latency_ms,
            actual_error_rate=actual_error_rate,
            labels=labels or {}
        )
        
        with self._lock:
            # Try to match to pending prediction
            for pred_id, pred in self.predictions.items():
                if (pred.outcome == PredictionOutcome.PENDING and 
                    pred.substrate_id == substrate_id):
                    # Check if prediction is still valid
                    elapsed = (sample.timestamp - pred.timestamp).total_seconds()
                    if elapsed <= pred.outcome_window_seconds:
                        sample.matched_prediction_id = pred_id
                        break
            
            self.ground_truth.append(sample)
        
        return sample.sample_id
    
    def validate_outcome(self, prediction_id: str, 
                        outcome: PredictionOutcome,
                        reason: Optional[str] = None) -> bool:
        """Manually validate a prediction outcome."""
        with self._lock:
            if prediction_id not in self.predictions:
                return False
            
            pred = self.predictions[prediction_id]
            pred.outcome = outcome
            pred.outcome_timestamp = datetime.now()
            pred.falsification_reason = reason
            
            # Update stats
            if outcome == PredictionOutcome.VALIDATED:
                self.validation_stats['validated'] += 1
            elif outcome == PredictionOutcome.FALSIFIED:
                self.validation_stats['falsified'] += 1
            elif outcome == PredictionOutcome.MISSED:
                self.validation_stats['missed'] += 1
            elif outcome == PredictionOutcome.TIMEOUT:
                self.validation_stats['timeouts'] += 1
        
        return True
    
    def get_falsification_rate(self) -> float:
        """Calculate falsification rate (false positive rate)."""
        with self._lock:
            completed = (self.validation_stats['validated'] + 
                        self.validation_stats['falsified'])
            if completed == 0:
                return 0.0
            return self.validation_stats['falsified'] / completed
    
    def get_pending_predictions(self) -> List[PredictionRecord]:
        """Get all predictions awaiting validation."""
        with self._lock:
            return [p for p in self.predictions.values() 
                   if p.outcome == PredictionOutcome.PENDING]
    
    def get_falsified_predictions(self) -> List[PredictionRecord]:
        """Get all falsified predictions for analysis."""
        with self._lock:
            return [p for p in self.predictions.values() 
                   if p.outcome == PredictionOutcome.FALSIFIED]


class Step3Gates:
    """
    Step 3 falsification gates.
    
    These gates use ground truth to validate Step 2 predictions:
    - precision: Of admitted requests, what fraction were actually stable?
    - recall: Of stable opportunities, what fraction did we admit?
    - calibration: Do predicted confidence match actual outcomes?
    """
    
    def __init__(self, ledger: TruthLedger):
        self.ledger = ledger
        
        # Default thresholds
        self.precision_threshold = 0.85  # 85% of admissions must succeed
        self.recall_threshold = 0.70     # Catch 70% of stable opportunities  
        self.calibration_threshold = 0.1  # |predicted - actual| < 10%
    
    def evaluate_precision(self, window_seconds: float = 300.0) -> dict:
        """
        Calculate precision: TP / (TP + FP)
        Of admitted requests, what fraction were actually stable?
        """
        with self.ledger._lock:
            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            
            completed = [
                p for p in self.ledger.predictions.values()
                if p.outcome in (PredictionOutcome.VALIDATED, PredictionOutcome.FALSIFIED)
                and p.timestamp > cutoff
            ]
            
            if not completed:
                return {'pass': True, 'value': 1.0, 'confidence': 0.0}
            
            validated = sum(1 for p in completed if p.outcome == PredictionOutcome.VALIDATED)
            precision = validated / len(completed)
            
            return {
                'pass': precision >= self.precision_threshold,
                'value': precision,
                'threshold': self.precision_threshold,
                'numerator': validated,
                'denominator': len(completed),
                'window_seconds': window_seconds
            }
    
    def evaluate_recall(self, window_seconds: float = 300.0) -> dict:
        """
        Calculate recall: TP / (TP + FN)
        Of stable opportunities, what fraction did we admit?
        
        Note: Requires ground truth on missed opportunities.
        """
        with self.ledger._lock:
            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            
            all_outcomes = [
                p for p in self.ledger.predictions.values()
                if p.timestamp > cutoff
                and p.outcome != PredictionOutcome.PENDING
            ]
            
            if not all_outcomes:
                return {'pass': True, 'value': 1.0, 'confidence': 0.0}
            
            validated = sum(1 for p in all_outcomes 
                          if p.outcome == PredictionOutcome.VALIDATED)
            missed = sum(1 for p in all_outcomes 
                        if p.outcome == PredictionOutcome.MISSED)
            
            denominator = validated + missed
            if denominator == 0:
                return {'pass': True, 'value': 1.0, 'confidence': 1.0}
            
            recall = validated / denominator
            
            return {
                'pass': recall >= self.recall_threshold,
                'value': recall,
                'threshold': self.recall_threshold,
                'validated': validated,
                'missed': missed,
                'window_seconds': window_seconds
            }
    
    def evaluate_calibration(self, bins: int = 5) -> dict:
        """
        Calculate calibration: Do predicted confidence match actual outcomes?
        
        Bin predictions by confidence, compare to actual success rate.
        """
        with self.ledger._lock:
            completed = [
                p for p in self.ledger.predictions.values()
                if p.outcome in (PredictionOutcome.VALIDATED, PredictionOutcome.FALSIFIED)
            ]
            
            if len(completed) < bins * 2:
                return {'pass': True, 'value': 1.0, 'calibration_error': 0.0}
            
            # Assign confidence based on gate metrics
            # Higher gate margin = higher confidence
            confidences = []
            outcomes = []
            
            for p in completed:
                # Simple confidence model: inverse of isolation
                confidence = max(0.0, min(1.0, 1.0 - p.isolation))
                confidences.append(confidence)
                outcomes.append(1.0 if p.outcome == PredictionOutcome.VALIDATED else 0.0)
            
            # Bin by confidence
            conf_arr = np.array(confidences)
            outcome_arr = np.array(outcomes)
            
            bin_edges = np.linspace(0, 1, bins + 1)
            calibration_errors = []
            
            for i in range(bins):
                mask = (conf_arr >= bin_edges[i]) & (conf_arr < bin_edges[i+1])
                if i == bins - 1:  # Include right edge
                    mask = (conf_arr >= bin_edges[i]) & (conf_arr <= bin_edges[i+1])
                
                if np.sum(mask) > 0:
                    bin_confidence = np.mean(conf_arr[mask])
                    bin_accuracy = np.mean(outcome_arr[mask])
                    calibration_errors.append(abs(bin_confidence - bin_accuracy))
            
            avg_calibration_error = np.mean(calibration_errors) if calibration_errors else 0.0
            
            return {
                'pass': avg_calibration_error <= self.calibration_threshold,
                'value': 1.0 - avg_calibration_error,
                'calibration_error': avg_calibration_error,
                'threshold': self.calibration_threshold,
                'n_samples': len(completed)
            }
    
    def evaluate_all(self) -> Dict[str, dict]:
        """Evaluate all Step 3 gates."""
        return {
            'precision': self.evaluate_precision(),
            'recall': self.evaluate_recall(),
            'calibration': self.evaluate_calibration(),
        }


class OutcomeTracker:
    """
    Tracks outcomes automatically by polling substrate state.
    
    Connects substrate measurements to truth ledger validation.
    """
    
    def __init__(
        self,
        ledger: TruthLedger,
        check_interval: float = 5.0,
        stability_threshold: float = 0.8,
        latency_threshold_ms: float = 100.0,
        error_rate_threshold: float = 0.01
    ):
        self.ledger = ledger
        self.check_interval = check_interval
        self.stability_threshold = stability_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._substrate_samplers: Dict[str, Callable[[], dict]] = {}
    
    def register_substrate(
        self,
        substrate_id: str,
        sampler: Callable[[], dict]
    ) -> None:
        """Register a function to sample substrate state."""
        self._substrate_samplers[substrate_id] = sampler
    
    def start(self) -> None:
        """Start outcome tracking."""
        self._running = True
        self._thread = threading.Thread(target=self._track_loop, daemon=True)
        self._thread.start()
        print(f"OutcomeTracker started (interval={self.check_interval}s)")
    
    def stop(self) -> None:
        """Stop tracking."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("OutcomeTracker stopped")
    
    def _track_loop(self) -> None:
        """Main tracking loop."""
        while self._running:
            self._check_all_outcomes()
            time.sleep(self.check_interval)
    
    def _check_all_outcomes(self) -> None:
        """Check outcomes for all pending predictions."""
        pending = self.ledger.get_pending_predictions()
        
        for pred in pending:
            if pred.substrate_id not in self._substrate_samplers:
                continue
            
            # Sample current substrate state
            try:
                sample = self._substrate_samplers[pred.substrate_id]()
            except Exception as e:
                continue
            
            # Record ground truth
            self.ledger.record_ground_truth(
                substrate_id=pred.substrate_id,
                actual_load=sample.get('load', 0.0),
                actual_stability=sample.get('stability', 0.0),
                actual_latency_ms=sample.get('latency_ms', 0.0),
                actual_error_rate=sample.get('error_rate', 0.0)
            )
            
            # Auto-validate based on thresholds
            elapsed = (datetime.now() - pred.timestamp).total_seconds()
            
            if elapsed > pred.outcome_window_seconds:
                # Window expired without falsification
                self.ledger.validate_outcome(
                    pred.prediction_id,
                    PredictionOutcome.VALIDATED,
                    "Window expired, no falsification detected"
                )
            else:
                # Check for falsification conditions
                if (sample.get('error_rate', 0) > self.error_rate_threshold or
                    sample.get('latency_ms', 0) > self.latency_threshold_ms):
                    self.ledger.validate_outcome(
                        pred.prediction_id,
                        PredictionOutcome.FALSIFIED,
                        f"Error rate: {sample.get('error_rate', 0):.3f}, "
                        f"Latency: {sample.get('latency_ms', 0):.1f}ms"
                    )


def create_truth_infrastructure(
    check_interval: float = 5.0
) -> tuple[TruthLedger, Step3Gates, OutcomeTracker]:
    """Factory: Create complete Step 3 truth infrastructure."""
    ledger = TruthLedger()
    gates = Step3Gates(ledger)
    tracker = OutcomeTracker(ledger, check_interval=check_interval)
    
    return ledger, gates, tracker
