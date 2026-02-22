# Capacity-Governed Systems Platform

A runtime for enforcing structural stability through capacity constraints.

## Quick Start

```python
from capacity_kernel import CapacityKernel, CapacityVector

kernel = CapacityKernel()
token = kernel.request_capacity(
    workload_id="my-service",
    C=CapacityVector(C_geo=0.7, C_int=0.5),
)

if token and kernel.check_capacity(token):
    # Do work within capacity bounds
    ...
```

## Architecture

- `capacity_kernel/` - Core runtime library
- `examples/` - Usage demonstrations
- `DESIGN.md` - Full design specification

## Testing

```bash
python examples/demo.py
```
