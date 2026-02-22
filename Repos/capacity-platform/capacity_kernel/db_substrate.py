"""Database connection pool substrate - different resource geometry.

This substrate treats a database connection pool as a discrete geometric space
where capacity governs concurrent query complexity.

Resource geometry:
- N_geo = connection pool size (discrete lattice points)
- C_geo = fraction of pool usable (geometric dimension budget)
- C_int = query interaction coupling (correlated vs independent queries)
- Gates measure pool pressure, query coherence, transaction isolation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import deque
import threading
import time
import uuid

import numpy as np


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    duration_ms: float
    rows_returned: int
    locks_held: int = 0
    waiting_for_lock: bool = False
    is_transaction: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConnectionPoolMetrics:
    """Snapshot of connection pool state."""
    pool_size: int
    active_connections: int
    idle_connections: int
    waiting_requests: int  # Queue depth
    total_queries: int
    slow_queries: int  # > 100ms
    deadlocked: int = 0
    
    @property
    def utilization(self) -> float:
        """Pool utilization fraction."""
        if self.pool_size == 0:
            return 0.0
        return self.active_connections / self.pool_size
    
    @property
    def contention(self) -> float:
        """Contention metric: waiting / (active + waiting)."""
        total = self.active_connections + self.waiting_requests
        if total == 0:
            return 0.0
        return self.waiting_requests / total


@dataclass
class QueryWindow:
    """Rolling window of query metrics for spectral analysis."""
    max_size: int = 64
    queries: deque = field(default_factory=lambda: deque(maxlen=64))
    durations: deque = field(default_factory=lambda: deque(maxlen=64))
    lock_events: deque = field(default_factory=lambda: deque(maxlen=64))
    
    def add_query(self, duration_ms: float, rows: int, locks: int = 0, 
                  waiting: bool = False) -> None:
        """Record a query execution."""
        self.queries.append({
            'duration': duration_ms,
            'rows': rows,
            'locks': locks,
            'waiting': waiting,
            'timestamp': datetime.now()
        })
        self.durations.append(duration_ms)
        self.lock_events.append(1 if waiting else 0)
    
    def to_spectrum(self) -> Dict[str, np.ndarray]:
        """Convert query patterns to spectral representation."""
        spectra = {}
        
        if len(self.durations) >= 8:
            dur_array = np.array(self.durations)
            # FFT of duration pattern (temporal frequency)
            fft = np.abs(np.fft.fft(dur_array - np.mean(dur_array)))
            spectra['duration'] = fft[:len(fft)//2]
            
            # Lock contention spectrum
            if len(self.lock_events) >= 8:
                lock_arr = np.array(self.lock_events)
                fft_lock = np.abs(np.fft.fft(lock_arr - np.mean(lock_arr)))
                spectra['contention'] = fft_lock[:len(fft_lock)//2]
        
        return spectra
    
    @property
    def dominant_frequency(self) -> float:
        """Dominant frequency in query pattern."""
        spectra = self.to_spectrum()
        if 'duration' in spectra and len(spectra['duration']) > 0:
            return float(np.max(spectra['duration']))
        return 0.0
    
    @property
    def query_interaction(self) -> float:
        """Measure of query interaction (λ_int analog)."""
        if len(self.durations) < 4:
            return 0.0
        
        dur_arr = np.array(list(self.durations))
        lock_arr = np.array(list(self.lock_events)) if self.lock_events else np.zeros(len(dur_arr))
        
        # Interaction = duration variance × lock correlation
        duration_variance = np.var(dur_arr) / (np.mean(dur_arr) ** 2 + 1e-10)
        lock_rate = np.mean(lock_arr) if len(lock_arr) > 0 else 0.0
        
        return duration_variance * (1 + lock_rate)


class DatabaseSubstrate:
    """
    Database connection pool substrate with discrete geometric structure.
    
    Unlike host substrate (continuous CPU/memory), this has:
    - Discrete N_geo = connection pool size
    - Boolean occupancy states (connection in use or not)
    - Query correlation patterns (temporal clustering)
    """
    
    def __init__(
        self,
        pool_size: int = 20,
        substrate_id: str = "postgres-main",
        poll_interval: float = 1.0
    ):
        self.pool_size = pool_size
        self.substrate_id = substrate_id
        self.poll_interval = poll_interval
        
        # Simulated pool state (in real impl, would connect to DB)
        self._active_conns: Dict[str, dict] = {}
        self._waiting_queue: deque = deque(maxlen=100)
        self._query_window = QueryWindow()
        
        self._metrics_history: deque = deque(maxlen=128)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._callbacks: List[Callable[[dict], None]] = []
    
    def _simulate_pool_activity(self) -> ConnectionPoolMetrics:
        """Simulate or measure real pool metrics."""
        # In real implementation, query pg_stat_activity or connection pool stats
        # For demo: simulate realistic patterns
        
        with self._lock:
            # Simulate some connections
            n_active = min(len(self._active_conns), self.pool_size)
            n_waiting = len(self._waiting_queue)
            
            metrics = ConnectionPoolMetrics(
                pool_size=self.pool_size,
                active_connections=n_active,
                idle_connections=self.pool_size - n_active,
                waiting_requests=n_waiting,
                total_queries=len(self._query_window.queries),
                slow_queries=sum(1 for q in self._query_window.queries 
                               if isinstance(q, dict) and q.get('duration', 0) > 100)
            )
            
            self._metrics_history.append(metrics)
            return metrics
    
    def _compute_gate_metrics(self, metrics: ConnectionPoolMetrics) -> dict:
        """Compute gate metrics for pool geometry."""
        
        # λ₁ - dominant frequency in query patterns
        lambda1 = self._query_window.dominant_frequency / 100.0
        
        # λ_int - interaction strength between queries
        lambda_int = self._query_window.query_interaction
        
        # Fit error - how well does current load match expected pattern
        if len(self._metrics_history) >= 4:
            utilizations = [m.utilization for m in list(self._metrics_history)[-4:]]
            trend = np.linspace(utilizations[0], utilizations[-1], len(utilizations))
            fit_error = float(np.std(np.array(utilizations) - trend))
        else:
            fit_error = 0.02
        
        # Gluing delta - coherence between connection states
        if len(self._query_window.queries) >= 8:
            recent = list(self._query_window.queries)[-8:]
            durations = [q.get('duration', 0) for q in recent]
            waiting = [1 if q.get('waiting', False) else 0 for q in recent]
            
            if np.std(durations) > 0.01 and np.std(waiting) > 0.01:
                coherence = np.abs(np.corrcoef(durations, waiting)[0, 1])
                gluing_delta = 1.0 - coherence if not np.isnan(coherence) else 0.001
            else:
                gluing_delta = 0.001
        else:
            gluing_delta = 0.001
        
        # Isolation metric - separation between query classes
        if len(self._query_window.queries) >= 8:
            # Measure separation between fast and slow queries
            durations = np.array([q.get('duration', 0) 
                                for q in list(self._query_window.queries)])
            if len(durations) > 1:
                # Cluster coefficient as isolation measure
                fast_mask = durations < np.percentile(durations, 50)
                slow_mask = ~fast_mask
                if np.any(fast_mask) and np.any(slow_mask):
                    fast_mean = np.mean(durations[fast_mask])
                    slow_mean = np.mean(durations[slow_mask])
                    total_var = np.var(durations)
                    between_var = abs(fast_mean - slow_mean) ** 2 / 4
                    isolation = between_var / (total_var + 1e-10) if total_var > 0 else 0.05
                else:
                    isolation = 0.05
            else:
                isolation = 0.05
        else:
            isolation = 0.05
        
        return {
            "fit_error": max(0.0, min(1.0, fit_error)),
            "gluing_delta": max(0.0, min(1.0, gluing_delta)),
            "lambda1": max(0.0, lambda1),
            "lambda_int": max(0.0, lambda_int),
            "isolation_metric": max(0.0, min(1.0, isolation)),
            "n_geo": self.pool_size ** 2,  # Discrete geometry
        }
    
    def measure_now(self) -> 'DBSubstrateState':
        """Take a measurement of the database substrate."""
        metrics = self._simulate_pool_activity()
        gate_metrics = self._compute_gate_metrics(metrics)
        
        state = DBSubstrateState(
            substrate_id=self.substrate_id,
            timestamp=datetime.now(),
            pool_metrics=metrics,
            **gate_metrics
        )
        
        return state
    
    def execute_query(self, duration_ms: float = 10.0, rows: int = 10,
                     acquire_lock: bool = False) -> str:
        """
        Track a query execution.
        
        Returns query ID if admitted, None if pool at capacity.
        """
        query_id = str(uuid.uuid4())[:8]
        
        with self._lock:
            if len(self._active_conns) >= self.pool_size:
                self._waiting_queue.append({
                    'query_id': query_id,
                    'timestamp': datetime.now()
                })
                self._query_window.add_query(duration_ms, rows, 
                                           locks=1 if acquire_lock else 0,
                                           waiting=True)
                return None  # Pool full
            
            self._active_conns[query_id] = {
                'started': datetime.now(),
                'duration': duration_ms
            }
            self._query_window.add_query(duration_ms, rows,
                                       locks=1 if acquire_lock else 0,
                                       waiting=False)
        
        return query_id
    
    def release_connection(self, query_id: str) -> None:
        """Release a connection back to the pool."""
        with self._lock:
            if query_id in self._active_conns:
                del self._active_conns[query_id]
            
            # Try to admit waiting queries
            while self._waiting_queue and len(self._active_conns) < self.pool_size:
                waiting = self._waiting_queue.popleft()
                self._active_conns[waiting['query_id']] = {
                    'started': datetime.now(),
                    'duration': 0  # Already waited
                }
    
    def start(self) -> None:
        """Start background monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _poll_loop(self) -> None:
        """Background measurement loop."""
        while self._running:
            state = self.measure_now()
            for cb in self._callbacks:
                try:
                    cb(state)
                except Exception as e:
                    print(f"Callback error: {e}")
            time.sleep(self.poll_interval)
    
    def on_measurement(self, callback: Callable[[dict], None]) -> None:
        """Register callback for measurements."""
        self._callbacks.append(callback)


@dataclass
class DBSubstrateState:
    """Complete substrate state for database connection pool."""
    substrate_id: str
    timestamp: datetime
    pool_metrics: ConnectionPoolMetrics
    fit_error: float
    gluing_delta: float
    lambda1: float
    lambda_int: float
    isolation_metric: float
    n_geo: int
    
    def to_spectral(self) -> Dict[str, np.ndarray]:
        """Convert to spectral representation."""
        # Pool geometry → eigenvalue analog
        utilization = self.pool_metrics.utilization
        contention = self.pool_metrics.contention
        
        return {
            "utilization_spectrum": np.array([utilization, contention]),
            "capacity_spectrum": np.array([self.lambda1, self.lambda_int]),
        }
