# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `mpitrampoline4jax`, a Python package that automatically configures MPITrampoline for JAX distributed computing. The package builds and bundles a C++ MPIWrapper library and sets the required environment variables when imported.

## Development Commands

- to build use ``uv build``. 
- to setup a clean environment ``uv venv``. 
- To run commands ``uv run python ...`` . 
-To install in editable mode ``uv sync``.

### Testing
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black .
uv run isort .
```

## Architecture

### Core Components

1. **Main Package (`src/mpitrampoline4jax/__init__.py`)**
   - Automatically sets `MPITRAMPOLINE_LIB` and `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` environment variables on import
   - Provides `is_configured()` and `get_library_path()` utility functions
   - Located at `src/mpitrampoline4jax/__init__.py:19-21`

2. **MPIWrapper Submodule (`src/MPIwrapper/`)**
   - C++ implementation that acts as an MPI trampoline/wrapper
   - Built using CMake and outputs `libmpiwrapper.so`
   - Companion to the MPItrampoline MPI implementation

3. **Custom Build System**
   - `setup.py` contains `BuildMPIWrapper` class that extends `build_ext`
   - Automatically patches CMakeLists.txt on macOS to disable two-level namespace checks
   - Builds library directly into package structure at `src/mpitrampoline4jax/lib/`

### Build Process

The build system performs these steps:
1. Creates build directory in `src/MPIwrapper/build/`
2. On macOS, patches CMakeLists.txt to comment out `check_twolevel.sh` validation
3. Runs CMake with output directory set to `src/mpitrampoline4jax/lib/`
4. Compiles the MPIWrapper C++ code to produce `libmpiwrapper.so`
5. Copies library to build directories for wheel packaging

### Usage Pattern

Users import the package before using JAX with MPI:
```python
import mpitrampoline4jax  # Sets environment variables
import jax
jax.distributed.initialize()
```

## Requirements

- CMake (for building MPIWrapper)
- A working MPI implementation (OpenMPI, MPICH, etc.)
- JAX >= 0.4
- Python >= 3.8

## Platform Notes

- **macOS**: Build system automatically patches CMakeLists.txt to disable strict two-level namespace checking
- **Library Extension**: Uses `.so` extension on all platforms (may need adjustment for Windows `.dll`)