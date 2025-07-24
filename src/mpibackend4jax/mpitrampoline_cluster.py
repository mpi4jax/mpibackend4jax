# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import socket
from jax._src import clusters

_PMI_SIZE = "PMI_SIZE"
_PMI_RANK = "PMI_RANK"
_MPITRAMPOLINE_ID = "MPITRAMPOLINE_ID"


class MPITrampolineLocalCluster(clusters.ClusterEnv):
    name: str = "mpitrampoline_local"

    @classmethod
    def is_env_present(cls) -> bool:
        return _PMI_SIZE in os.environ and _PMI_RANK in os.environ

    @classmethod
    def get_coordinator_address(cls, timeout_secs: int | None) -> str:
        # Try to get a more robust coordinator address
        # First, check if MPITRAMPOLINE_ID is available for port generation
        if _MPITRAMPOLINE_ID in os.environ:
            # Use MPITRAMPOLINE_ID to generate a deterministic port
            job_id = int(os.environ[_MPITRAMPOLINE_ID])
            port = 40000 + (job_id % (2**12))  # Port in range 40000-44095
        else:
            # Fallback to a fixed port if no job ID available
            port = 50000

        # Try to determine the coordinator host
        # Check common MPI environment variables for hostname
        hostname = "127.0.0.1"  # Default fallback

        # Try various MPI environment variables that might contain hostname info
        for env_var in [
            "HYDRA_HOST_FILE",
            "MPI_LOCALNRANKS",
            "OMPI_MCA_orte_local_daemon_uri",
        ]:
            if env_var in os.environ:
                # For now, use localhost as we can't easily parse these
                # In a real deployment, rank 0 would typically be the coordinator
                break

        # If we're rank 0, we're likely the coordinator, so use actual hostname
        if os.environ.get(_PMI_RANK, "0") == "0":
            try:
                hostname = socket.gethostname()
                # Try to resolve to IP address for better reliability
                hostname = socket.gethostbyname(hostname)
            except (socket.gaierror, OSError):
                hostname = "127.0.0.1"  # Fallback on error

        return f"{hostname}:{port}"

    @classmethod
    def get_process_count(cls) -> int:
        return int(os.environ[_PMI_SIZE])

    @classmethod
    def get_process_id(cls) -> int:
        return int(os.environ[_PMI_RANK])

    @classmethod
    def get_local_process_id(cls) -> int | None:
        return int(os.environ[_PMI_RANK])
