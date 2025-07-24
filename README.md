# mpitrampoline4jax

A Python package that provides MPITrampoline integration for JAX, automatically building MPIWrapper and setting required environment variables.

## Installation

```bash
pip install .
```

## Usage

Simply import the package before using JAX with MPI:

```python
import mpitrampoline4jax
import jax

# Initialize JAX distributed
jax.distributed.initialize()

# Your JAX code here
```

Or run directly:

```bash
mpirun -np 2 python -c 'import mpitrampoline4jax; import jax; jax.distributed.initialize()'
```

## What it does

When you import `mpitrampoline4jax`, it automatically:

1. Sets `MPITRAMPOLINE_LIB` to point to the built `libmpiwrapper.so`
2. Sets `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`

## Requirements

- CMake (for building MPIWrapper)
- A working MPI implementation (e.g., OpenMPI, MPICH)
- JAX

## Verification

You can check if MPITrampoline is properly configured:

```python
import mpitrampoline4jax

if mpitrampoline4jax.is_configured():
    print("MPITrampoline is properly configured!")
else:
    print("MPITrampoline configuration failed.")
```