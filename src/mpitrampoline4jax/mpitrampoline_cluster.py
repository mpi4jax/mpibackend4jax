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
from jax._src import clusters

_PMI_SIZE = "PMI_SIZE"
_PMI_RANK = "PMI_RANK"


class MPITrampolineLocalCluster(clusters.ClusterEnv):
    name: str = "mpitrampoline_local"

    @classmethod
    def is_env_present(cls) -> bool:
        return _PMI_SIZE in os.environ and _PMI_RANK in os.environ

    @classmethod
    def get_coordinator_address(cls, timeout_secs: int | None) -> str:
        return "127.0.0.1:50000"

    @classmethod
    def get_process_count(cls) -> int:
        return int(os.environ[_PMI_SIZE])

    @classmethod
    def get_process_id(cls) -> int:
        return int(os.environ[_PMI_RANK])

    @classmethod
    def get_local_process_id(cls) -> int | None:
        return int(os.environ[_PMI_RANK])
