"""GPU Tensor Core substrate - matrix compute geometry.

This substrate treats GPU resources as a tensor-product space where
capacity governs matrix operation complexity.

Resource geometry:
- N_geo = SMs × tensor_cores × warp_size (discrete 3D lattice)
- C_geo = fraction of tensor cores usable (matrix dimension budget)
- C_int = operation pipelining (overlapped vs sequential ops)
- Gates measure occupancy, kernel coherence, memory/compute balance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime
from collections import deque
import threading
import time
import uuid

import numpy as np


@dataclass
class KernelMetrics:
    """Metrics for a GPU kernel execution."""
    kernel_id: str
    duration_ms: float
    grid_size: Tuple[int, int, int]  # Blocks per dimension
    block_size: Tuple[int, int, int]  # Threads per block
    sm_occupancy: float  # 0-1
    shared_mem_kb: int
    registers_per_thread: int
    is_tensor_op: bool  # Uses tensor cores (GEMM, conv)
    memory_bw_gbps: float
    compute_util: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GPUDeviceMetrics:
    """Snapshot of GPU device state."""
    device_id: str
    sm_count: int
    tensor_cores: int
    memory_total_gb: float
    memory_used_gb: float
    
    # Runtime metrics
    active_kernels: int
    queue_depth: int
    avg_sm_occupancy: float
    tensor_core_util: float  # Fraction of tensor cores active
    memory_bw_util: float  # Fraction of peak bandwidth
    temperature: float
    power_watts: float
    
    @property
    def memory_util(self) -> float:
        """Memory utilization fraction."""
        if self.memory_total_gb == 0:
            return 0.0
        return self.memory_used_gb / self.memory_total_gb
    
    @property
    def compute_pressure(self) -> float:
        """Compute pressure: active kernels / SMs."""
        if self.sm_count == 0:
            return 0.0
        return min(1.0, self.active_kernels / self.sm_count)


@dataclass
class KernelWindow:
    """Rolling window of kernel executions for spectral analysis."""
    max_size: int = 64
    kernels: deque = field(default_factory=lambda: deque(maxlen=64))
    durations: deque = field(default_factory=lambda: deque(maxlen=64))
    occupancies: deque = field(default_factory=lambda: deque(maxlen=64))
    
    def add_kernel(self, duration_ms: float, sm_occupancy: float,
                   is_tensor_op: bool, memory_bw: float) -> None:
        """Record a kernel execution."""
        self.kernels.append({
            'duration': duration_ms,
            'occupancy': sm_occupancy,
            'tensor_op': 1 if is_tensor_op else 0,
            'memory_bw': memory_bw,
            'timestamp': datetime.now()
        })
        self.durations.append(duration_ms)
        self.occupancies.append(sm_occupancy)
    
    def to_spectrum(self) -> Dict[str, np.ndarray]:
        """Convert kernel patterns to spectral representation."""
        spectra = {}
        
        if len(self.durations) >= 8:
            dur_array = np.array(self.durations)
            # FFT of kernel duration pattern
            fft = np.abs(np.fft.fft(dur_array - np.mean(dur_array)))
            spectra['duration'] = fft[:len(fft)//2]
            
            # Occupancy pattern spectrum
            occ_array = np.array(self.occupancies)
            fft_occ = np.abs(np.fft.fft(occ_array - np.mean(occ_array)))
            spectra['occupancy'] = fft_occ[:len(fft_occ)//2]
            
            # Tensor op spectrum (binary pattern)
            tensor_arr = np.array([k.get('tensor_op', 0) for k in self.kernels])
            if len(tensor_arr) >= 8:
                fft_tensor = np.abs(np.fft.fft(tensor_arr - np.mean(tensor_arr)))
                spectra['tensor_ops'] = fft_tensor[:len(fft_tensor)//2]
        
        return spectra
    
    @property
    def dominant_frequency(self) -> float:
        """Dominant frequency in kernel execution pattern."""
        spectra = self.to_spectrum()
        if 'occupancy' in spectra and len(spectra['occupancy']) > 0:
            return float(np.max(spectra['occupancy']))
        return 0.0
    
    @property
    def pipeline_interaction(self) -> float:
        """Measure of kernel interaction / pipelining efficiency."""
        if len(self.kernels) < 4:
            return 0.0
        
        # Interaction = occupancy variance × tensor op density
        occ_arr = np.array(list(self.occupancies))
        tensor_arr = np.array([k.get('tensor_op', 0) for k in self.kernels])
        
        occupancy_variance = np.var(occ_arr)
        tensor_density = np.mean(tensor_arr)
        
        # High occupancy variance + high tensor density = strained interaction
        return occupancy_variance * (1 + tensor_density * 2)


class TensorCoreSubstrate:
    """
    GPU Tensor Core substrate with 3D compute geometry.
    
    Geometry: N_geo = SM_count × tensor_cores × warp_size
    
    Unlike host (continuous) or DB (2D discrete), this has:
    - 3D lattice: SMs × tensor_cores × warps
    - Hierarchical capacity: SM occupancy + tensor core availability
    - Compute/memory balance gates
    """
    
    def __init__(
        self,
        sm_count: int = 80,
        tensor_cores_per_sm: int = 4,
        warp_size: int = 32,
        device_id: str = "cuda:0",
        poll_interval: float = 0.5
    ):
        self.sm_count = sm_count
        self.tensor_cores_per_sm = tensor_cores_per_sm
        self.warp_size = warp_size
        self.device_id = device_id
        self.poll_interval = poll_interval
        
        # Simulated GPU state (in real impl, would use nvml or CUDA APIs)
        self._active_kernels: Dict[str, KernelMetrics] = {}
        self._kernel_queue: deque = deque(maxlen=128)
        self._kernel_window = KernelWindow()
        
        self._metrics_history: deque = deque(maxlen=128)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._callbacks: List[Callable[[dict], None]] = []
        
        # Simulated device state
        self._memory_total = 80.0  # GB (A100-like)
        self._memory_used = 0.0
        self._temperature = 45.0
        self._power = 150.0
    
    @property
    def n_geo(self) -> int:
        """Total geometric degrees of freedom."""
        return self.sm_count * self.tensor_cores_per_sm * self.warp_size
    
    def _simulate_gpu_state(self) -> GPUDeviceMetrics:
        """Simulate or measure real GPU metrics."""
        # In real implementation, query nvmlDeviceGetUtilizationRates, etc.
        # For demo: simulate realistic patterns based on active kernels
        
        with self._lock:
            n_active = len(self._active_kernels)
            n_queued = len(self._kernel_queue)
            
            # Simulate memory growth/decay
            target_memory = min(self._memory_total, n_active * 0.5)  # ~0.5GB per kernel
            self._memory_used = 0.9 * self._memory_used + 0.1 * target_memory
            
            # Simulate thermals
            load_factor = n_active / self.sm_count
            target_temp = 45.0 + load_factor * 40.0  # 45-85°C range
            self._temperature = 0.95 * self._temperature + 0.05 * target_temp
            
            # Tensor core utilization from active kernels
            tensor_ops_active = sum(1 for k in self._active_kernels.values() 
                                   if k.is_tensor_op)
            total_tensor_cores = self.sm_count * self.tensor_cores_per_sm
            tensor_util = tensor_ops_active * self.tensor_cores_per_sm / total_tensor_cores
            
            avg_occupancy = 0.0
            if self._active_kernels:
                avg_occupancy = sum(k.sm_occupancy for k in self._active_kernels.values())
                avg_occupancy /= len(self._active_kernels)
            
            metrics = GPUDeviceMetrics(
                device_id=self.device_id,
                sm_count=self.sm_count,
                tensor_cores=self.sm_count * self.tensor_cores_per_sm,
                memory_total_gb=self._memory_total,
                memory_used_gb=self._memory_used,
                active_kernels=n_active,
                queue_depth=n_queued,
                avg_sm_occupancy=avg_occupancy,
                tensor_core_util=tensor_util,
                memory_bw_util=load_factor * 0.6,  # Simulated
                temperature=self._temperature,
                power_watts=self._power + load_factor * 200.0
            )
            
            self._metrics_history.append(metrics)
            return metrics
    
    def _compute_gate_metrics(self, metrics: GPUDeviceMetrics) -> dict:
        """Compute gate metrics for tensor core geometry."""
        
        # λ₁ - dominant frequency in SM occupancy pattern
        lambda1 = self._kernel_window.dominant_frequency / 100.0
        
        # λ_int - pipeline interaction (how kernels interfere)
        lambda_int = self._kernel_window.pipeline_interaction
        
        # Fit error - deviation from ideal occupancy pattern
        if len(self._metrics_history) >= 8:
            occupancies = [m.avg_sm_occupancy for m in list(self._metrics_history)[-8:]]
            # Ideal occupancy varies with compute load
            ideal = np.linspace(occupancies[0], occupancies[-1], len(occupancies))
            fit_error = float(np.std(np.array(occupancies) - ideal))
        else:
            fit_error = 0.02
        
        # Gluing delta - coherence between SM occupancy and memory bandwidth
        if len(self._kernel_window.kernels) >= 8:
            recent = list(self._kernel_window.kernels)[-8:]
            occupancies = [k.get('occupancy', 0) for k in recent]
            memory_bws = [k.get('memory_bw', 0) for k in recent]
            
            if np.std(occupancies) > 0.001 and np.std(memory_bws) > 0.001:
                coherence = np.abs(np.corrcoef(occupancies, memory_bws)[0, 1])
                gluing_delta = 1.0 - coherence if not np.isnan(coherence) else 0.001
            else:
                gluing_delta = 0.001
        else:
            gluing_delta = 0.001
        
        # Isolation metric - separation between tensor and non-tensor ops
        if len(self._kernel_window.kernels) >= 8:
            tensor_ops = np.array([k.get('tensor_op', 0) for k in list(self._kernel_window.kernels)])
            other_ops = 1 - tensor_ops
            
            if np.any(tensor_ops == 1) and np.any(tensor_ops == 0):
                # Measure separation between tensor and non-tensor phases
                tensor_mask = tensor_ops == 1
                tensor_ratio = np.mean(tensor_ops)
                
                # Isolation = |tensor_ratio - 0.5| × 2 (0 = balanced, 1 = isolated)
                isolation = abs(tensor_ratio - 0.5) * 2
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
            "n_geo": self.n_geo,
        }
    
    def measure_now(self) -> 'TCoreSubstrateState':
        """Take a measurement of the GPU substrate."""
        metrics = self._simulate_gpu_state()
        gate_metrics = self._compute_gate_metrics(metrics)
        
        state = TCoreSubstrateState(
            substrate_id=self.device_id,
            timestamp=datetime.now(),
            gpu_metrics=metrics,
            **gate_metrics
        )
        
        return state
    
    def launch_kernel(self, grid: Tuple[int, int, int], 
                     block: Tuple[int, int, int],
                     sm_occupancy: float = 0.8,
                     is_tensor_op: bool = True,
                     shared_mem_kb: int = 48) -> Optional[str]:
        """
        Track a kernel launch. Returns kernel ID if admitted.
        """
        kernel_id = str(uuid.uuid4())[:8]
        
        # Calculate resource requirements
        threads_per_block = block[0] * block[1] * block[2]
        total_threads = threads_per_block * grid[0] * grid[1] * grid[2]
        warps_needed = total_threads / self.warp_size
        sms_needed = max(1, int(warps_needed / 32))  # Rough estimate
        
        with self._lock:
            # Check SM availability
            sms_in_use = sum(
                max(1, int(k.grid_size[0] * k.grid_size[1] * k.grid_size[2] * 
                          k.block_size[0] * k.block_size[1] * k.block_size[2] / 
                          (self.warp_size * 32 * 32)))
                for k in self._active_kernels.values()
            )
            
            if sms_in_use + sms_needed > self.sm_count:
                self._kernel_queue.append({
                    'kernel_id': kernel_id,
                    'timestamp': datetime.now()
                })
                # Still track for metrics
                duration = 10.0 + np.random.exponential(20.0)  # ms
                self._kernel_window.add_kernel(duration, 0.0, is_tensor_op, 50.0)
                return None  # Queued
            
            # Launch on device
            duration = 10.0 + np.random.exponential(10.0)  # ms
            
            kernel = KernelMetrics(
                kernel_id=kernel_id,
                duration_ms=duration,
                grid_size=grid,
                block_size=block,
                sm_occupancy=sm_occupancy,
                shared_mem_kb=shared_mem_kb,
                registers_per_thread=64,
                is_tensor_op=is_tensor_op,
                memory_bw_gbps=200.0 + np.random.exponential(100.0),
                compute_util=sm_occupancy * (1.5 if is_tensor_op else 1.0)
            )
            
            self._active_kernels[kernel_id] = kernel
            
            self._kernel_window.add_kernel(
                duration, sm_occupancy, is_tensor_op, kernel.memory_bw_gbps
            )
        
        return kernel_id
    
    def synchronize(self, kernel_id: str) -> None:
        """Mark kernel as complete."""
        with self._lock:
            if kernel_id in self._active_kernels:
                del self._active_kernels[kernel_id]
            
            # Try to admit queued kernels
            while self._kernel_queue and len(self._active_kernels) < self.sm_count:
                queued = self._kernel_queue.popleft()
                # In real impl, would actually launch
    
    def start(self) -> None:
        """Start background monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"TensorCoreSubstrate[{self.device_id}] started")
    
    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"TensorCoreSubstrate[{self.device_id}] stopped")
    
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
class TCoreSubstrateState:
    """Complete substrate state for tensor core GPU."""
    substrate_id: str
    timestamp: datetime
    gpu_metrics: GPUDeviceMetrics
    fit_error: float
    gluing_delta: float
    lambda1: float
    lambda_int: float
    isolation_metric: float
    n_geo: int
    
    def to_spectral(self) -> Dict[str, np.ndarray]:
        """Convert to spectral representation."""
        return {
            "occupancy_spectrum": np.array([
                self.gpu_metrics.avg_sm_occupancy,
                self.gpu_metrics.tensor_core_util
            ]),
            "capacity_spectrum": np.array([self.lambda1, self.lambda_int]),
        }
