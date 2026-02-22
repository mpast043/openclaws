"""Real-time substrate measurement and monitoring."""

import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, List
from datetime import datetime
from collections import deque
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .substrate import SubstrateState, ResourceMetrics, ServiceMetrics


@dataclass
class MeasurementWindow:
    """Rolling window of measurements for spectral analysis."""
    max_size: int = 64
    cpu_samples: deque = field(default_factory=lambda: deque(maxlen=64))
    memory_samples: deque = field(default_factory=lambda: deque(maxlen=64))
    io_samples: deque = field(default_factory=lambda: deque(maxlen=64))
    
    def add(self, cpu: float, memory: float, io: float) -> None:
        """Add a new sample to the window."""
        self.cpu_samples.append(cpu)
        self.memory_samples.append(memory)
        self.io_samples.append(io)
    
    def to_spectrum(self) -> dict:
        """Convert samples to spectral representation (eigenvalue analog)."""
        spectra = {}
        
        if len(self.cpu_samples) >= 4:
            # Use FFT on CPU samples as spectral analog
            cpu_array = np.array(self.cpu_samples)
            fft = np.abs(np.fft.fft(cpu_array - np.mean(cpu_array)))
            spectra['cpu'] = fft[:len(fft)//2]  # Positive frequencies only
        
        if len(self.memory_samples) >= 4:
            mem_array = np.array(self.memory_samples)
            fft = np.abs(np.fft.fft(mem_array - np.mean(mem_array)))
            spectra['memory'] = fft[:len(fft)//2]
        
        return spectra
    
    @property
    def dominant_frequency(self) -> float:
        """Extract dominant frequency (analog to λ₁ in eigenvalue spectrum)."""
        spectra = self.to_spectrum()
        if 'cpu' in spectra and len(spectra['cpu']) > 0:
            return float(np.max(spectra['cpu']))
        return 0.0
    
    @property
    def interaction_strength(self) -> float:
        """Compute interaction strength across resources (λ_int analog)."""
        if len(self.cpu_samples) < 4:
            return 0.0
        
        cpu_arr = np.array(list(self.cpu_samples))
        mem_arr = np.array(list(self.memory_samples)) if self.memory_samples else cpu_arr * 0.5
        
        # Interaction = correlation between CPU and memory pressure
        if len(cpu_arr) == len(mem_arr) and len(cpu_arr) > 1:
            correlation = np.corrcoef(cpu_arr, mem_arr)[0, 1]
            # Handle NaN case (no variance in one variable)
            if not np.isnan(correlation):
                return abs(correlation) * np.mean(cpu_arr + mem_arr) / 2.0
        return np.mean(cpu_arr)


class SubstrateMonitor:
    """
    Real-time substrate measurement system.
    
    Polls actual system resources and maintains spectral windows
    for gate evaluation.
    """
    
    def __init__(
        self,
        substrate_id: str = "default",
        poll_interval: float = 1.0,
        window_size: int = 64
    ):
        self.substrate_id = substrate_id
        self.poll_interval = poll_interval
        self.window = MeasurementWindow(max_size=window_size)
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[SubstrateState], None]] = []
        self._latest_state: Optional[SubstrateState] = None
        self._lock = threading.Lock()
        
        # For I/O rate calculation
        self._last_io_counters = None
        self._last_io_time = None
    
    def _measure_resources(self) -> ResourceMetrics:
        """Take real system measurements using psutil."""
        if not PSUTIL_AVAILABLE:
            # Fallback to sample data if psutil not installed
            return ResourceMetrics(
                cpu_percent=45.0,
                memory_percent=60.0,
                network_io_mbps=100.0,
                disk_io_mbps=50.0,
            )
        
        # CPU measurement
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory measurement
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Network I/O rate calculation
        net_io = psutil.net_io_counters()
        network_mbps = 0.0
        if self._last_io_counters is not None:
            bytes_delta = (net_io.bytes_sent + net_io.bytes_recv) - (
                self._last_io_counters.bytes_sent + self._last_io_counters.bytes_recv
            )
            time_delta = time.time() - self._last_io_time
            if time_delta > 0:
                network_mbps = (bytes_delta * 8 / 1e6) / time_delta
        self._last_io_counters = net_io
        self._last_io_time = time.time()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_mbps = 0.0
        if disk_io:
            disk_mbps = (disk_io.read_bytes + disk_io.write_bytes) / 1e6
        
        # Build spectral analogs from window
        spectra = self.window.to_spectrum()
        cpu_spectrum = spectra.get('cpu', np.array([cpu_percent]))
        mem_spectrum = spectra.get('memory', np.array([memory_percent]))
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            network_io_mbps=network_mbps,
            disk_io_mbps=disk_mbps,
            cpu_load_spectrum=cpu_spectrum,
            memory_pressure_spectrum=mem_spectrum,
        )
    
    def _compute_gate_metrics(self) -> dict:
        """Compute metrics for gate evaluation from measurement window."""
        # These are the "physics" of the substrate
        
        # λ₁ - smallest significant eigenvalue (dominant frequency)
        lambda1 = self.window.dominant_frequency / 100.0  # Normalize
        
        # λ_int - interaction strength between modes
        lambda_int = self.window.interaction_strength
        
        # Fit error - deviation from expected behavior (normalized std of detrended signal)
        if len(self.window.cpu_samples) >= 8:
            cpu_arr = np.array(list(self.window.cpu_samples))
            # Fit error = variance of deviations from trend
            trend = np.linspace(cpu_arr[0], cpu_arr[-1], len(cpu_arr))
            fit_error = float(np.std(cpu_arr - trend)) / 100.0
        else:
            # Assume good fit with limited data
            fit_error = 0.02
        
        # Gluing delta - coherence between measurements
        if len(self.window.cpu_samples) >= 8 and len(self.window.memory_samples) >= 8:
            cpu_last = np.array(list(self.window.cpu_samples)[-8:])
            mem_last = np.array(list(self.window.memory_samples)[-8:])
            # Gluing = how well CPU/Mem measurements cohere (high coherence = low delta)
            if np.std(cpu_last) > 0.01 and np.std(mem_last) > 0.01:
                coherence = np.abs(np.corrcoef(cpu_last, mem_last)[0, 1])
                gluing_delta = 1.0 - coherence
            else:
                # Low variance = high coherence by default
                gluing_delta = 0.001
        else:
            # Not enough samples - assume high coherence (low delta)
            gluing_delta = 0.001
        
        # Isolation metric - cross-contamination between resources
        spectra = self.window.to_spectrum()
        if 'cpu' in spectra and 'memory' in spectra and len(spectra['cpu']) >= 4:
            # Isolation = spectral overlap (similarity of spectra)
            cpu_spec = spectra['cpu']
            mem_spec = spectra['memory']
            min_len = min(len(cpu_spec), len(mem_spec))
            if min_len > 0:
                cpu_norm = cpu_spec[:min_len] / (np.linalg.norm(cpu_spec[:min_len]) + 1e-10)
                mem_norm = mem_spec[:min_len] / (np.linalg.norm(mem_spec[:min_len]) + 1e-10)
                isolation = float(np.abs(np.dot(cpu_norm, mem_norm)))
            else:
                isolation = 0.05
        else:
            # Not enough spectral data - assume good isolation
            isolation = 0.05
        
        return {
            "fit_error": max(0.0, min(1.0, fit_error)),
            "gluing_delta": max(0.0, min(1.0, gluing_delta)),
            "lambda1": max(0.0, lambda1),
            "lambda_int": max(0.0, lambda_int),
            "isolation_metric": max(0.0, min(1.0, isolation)),
        }
    
    def measure_now(self) -> SubstrateState:
        """Take a single measurement and return substrate state."""
        resources = self._measure_resources()
        
        # Add to window for spectral analysis
        self.window.add(
            cpu=resources.cpu_percent,
            memory=resources.memory_percent,
            io=resources.network_io_mbps + resources.disk_io_mbps
        )
        
        gate_metrics = self._compute_gate_metrics()
        
        state = SubstrateState(
            substrate_id=self.substrate_id,
            timestamp=datetime.now(),
            resources=resources,
            services={},  # TODO: service discovery
            **gate_metrics,
            n_geo=64**3,  # Resolution
        )
        
        with self._lock:
            self._latest_state = state
        
        return state
    
    def _poll_loop(self) -> None:
        """Background polling thread."""
        while self._running:
            state = self.measure_now()
            
            # Notify callbacks
            for cb in self._callbacks:
                try:
                    cb(state)
                except Exception as e:
                    print(f"Callback error: {e}")
            
            time.sleep(self.poll_interval)
    
    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"SubstrateMonitor[{self.substrate_id}] started (poll={self.poll_interval}s)")
    
    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print(f"SubstrateMonitor[{self.substrate_id}] stopped")
    
    @property
    def current_state(self) -> Optional[SubstrateState]:
        """Get latest measured state."""
        with self._lock:
            return self._latest_state
    
    def on_measurement(self, callback: Callable[[SubstrateState], None]) -> None:
        """Register a callback for each measurement."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[SubstrateState], None]) -> None:
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)


def create_real_substrate(substrate_id: str = "host") -> SubstrateState:
    """
    One-shot real substrate measurement.
    
    Usage:
        state = create_real_substrate()
        print(f"CPU: {state.resources.cpu_percent}%")
    """
    monitor = SubstrateMonitor(substrate_id=substrate_id, poll_interval=1.0)
    return monitor.measure_now()
