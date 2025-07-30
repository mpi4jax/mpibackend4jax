import os
import warnings

from pathlib import Path

import jax


def initialize():
    # Get the package installation directory
    _package_dir = Path(__file__).parent
    _mpiwrapper_lib = _package_dir / "lib" / "libmpiwrapper.so"

    if jax.initialized.is_initialized():
        warnings.warn_explicit(
            "JAX is already initialized. You must load mpibackend4jax before initializing JAX."
        )

    # Set environment variables for MPITrampoline
    if _mpiwrapper_lib.exists():
        if "MPITRAMPOLINE_LIB" not in os.environ.keys():
            os.environ["MPITRAMPOLINE_LIB"] = str(_mpiwrapper_lib.absolute())
            print(f"mpibackend4jax: Set MPITRAMPOLINE_LIB={_mpiwrapper_lib.absolute()}")
        if "JAX_CPU_COLLECTIVES_IMPLEMENTATION" not in os.environ.keys():
            os.environ["JAX_CPU_COLLECTIVES_IMPLEMENTATION"] = "mpi"
            print("mpibackend4jax: Set JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi")
    else:
        print(f"Warning: MPIWrapper library not found at {_mpiwrapper_lib}")
        print("Please ensure the package was installed correctly.")
