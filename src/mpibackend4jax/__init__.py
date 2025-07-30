"""
mpibackend4jax - MPI backend integration for JAX

This package automatically configures MPITrampoline for use with JAX
by setting the required environment variables when imported.
"""

import os

# Import the cluster to register it automatically
from .mpitrampoline_cluster import MPITrampolineLocalCluster

__version__ = "0.1.0"


# Convenience function to check if MPITrampoline is properly configured
def is_configured():
    """Check if MPITrampoline is properly configured for JAX"""
    return (
        "MPITRAMPOLINE_LIB" in os.environ
        and os.environ.get("JAX_CPU_COLLECTIVES_IMPLEMENTATION") == "mpi"
        and Path(os.environ["MPITRAMPOLINE_LIB"]).exists()
    )


def get_library_path():
    """Get the path to the MPIWrapper library"""
    return os.environ.get("MPITRAMPOLINE_LIB")


__all__ = ["is_configured", "get_library_path", "MPITrampolineLocalCluster"]
