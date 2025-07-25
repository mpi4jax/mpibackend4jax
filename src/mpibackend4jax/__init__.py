"""
mpibackend4jax - MPI backend integration for JAX

This package automatically configures MPITrampoline for use with JAX
by setting the required environment variables when imported.
"""

import os
from pathlib import Path

# Import the cluster to register it automatically
# from .mpitrampoline_cluster import MPITrampolineLocalCluster

__version__ = "0.1.0"

def initialize():
    # Get the package installation directory
    _package_dir = Path(__file__).parent
    _mpiwrapper_lib = _package_dir / "lib" / "libmpiwrapper.so"

    # Set environment variables for MPITrampoline
    if _mpiwrapper_lib.exists():
        os.environ["MPITRAMPOLINE_LIB"] = str(_mpiwrapper_lib.absolute())
        os.environ["JAX_CPU_COLLECTIVES_IMPLEMENTATION"] = "mpi"

        print(f"mpibackend4jax: Set MPITRAMPOLINE_LIB={_mpiwrapper_lib.absolute()}")
        print("mpibackend4jax: Set JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi")
    else:
        print(f"Warning: MPIWrapper library not found at {_mpiwrapper_lib}")
        print("Please ensure the package was installed correctly.")


# # Convenience function to check if MPITrampoline is properly configured
# def is_configured():
#     """Check if MPITrampoline is properly configured for JAX"""
#     return (
#         "MPITRAMPOLINE_LIB" in os.environ
#         and os.environ.get("JAX_CPU_COLLECTIVES_IMPLEMENTATION") == "mpi"
#         and Path(os.environ["MPITRAMPOLINE_LIB"]).exists()
#     )
#
#
# def get_library_path():
#     """Get the path to the MPIWrapper library"""
#     return os.environ.get("MPITRAMPOLINE_LIB")


__all__ = ["is_configured", "get_library_path"]#, "MPITrampolineLocalCluster"]
