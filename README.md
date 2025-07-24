# mpitrampoline4jax

> [!WARNING]
> This is an early experimental package. Feedback wanted!

A Python package to provide an easy to use MPI-backend for JAX sharding, built on top of MPIWrapper. No special operations, 100% native JAX.

## Installation

```bash
# Using uv (recommended)
uv add git+https://github.com/mpi4jax/mpitrampoline4jax

# Using pip
pip install git+https://github.com/mpi4jax/mpitrampoline4jax
```

## Usage

Simply import the package before using JAX with MPI:

```python
import mpitrampoline4jax as _mpi4jax  # noqa: F401
import jax

print("Setup initialize", flush=True)
jax.distributed.initialize()
print(f"{jax.process_index()}/{jax.process_count()} :", jax.local_devices())
print(f"{jax.process_index()}/{jax.process_count()} :", jax.devices())

x = jax.numpy.ones(
    (jax.device_count(),),
    device=jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), "i"), jax.sharding.PartitionSpec("i")
    ),
)

print(f"{jax.process_index()}/{jax.process_count()} :", x.sum())
```

Run with MPI:

```bash
mpirun -np 2 python examples/example.py
```

## What it does

When you import `mpitrampoline4jax`, it automatically:

1. Sets `MPITRAMPOLINE_LIB` to point to the built `libmpiwrapper.so`
2. Sets `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`

## Requirements

- CMake (for building MPIWrapper)
- A working MPI implementation (e.g., OpenMPI, MPICH)
- JAX

Tested on macOS with MPICH.

## Verification

You can check if MPITrampoline is properly configured:

```python
import mpitrampoline4jax

if mpitrampoline4jax.is_configured():
    print("MPITrampoline is properly configured!")
else:
    print("MPITrampoline configuration failed.")
```

## Acknowledgments

Special thanks to @inailuig (Clemens Giuliani) for adding MPI support in XLA, which makes this integration possible.