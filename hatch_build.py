"""Hatch build hook for compiling MPIWrapper with CMake."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to compile MPIWrapper using CMake."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        """Initialize the build hook and compile MPIWrapper."""
        if self.target_name == "sdist":
            return

        self._build_mpiwrapper()

        # For wheel builds, ensure the lib directory is included
        if self.target_name == "wheel":
            if "force_include" not in build_data:
                build_data["force_include"] = {}
            build_data["force_include"]["src/mpibackend4jax/lib"] = "mpibackend4jax/lib"

    def _build_mpiwrapper(self) -> None:
        """Build MPIWrapper using CMake."""
        root_dir = Path(self.root)
        mpiwrapper_dir = root_dir / "src" / "MPIwrapper"
        package_lib_dir = root_dir / "src" / "mpibackend4jax" / "lib"

        # Create lib directory
        package_lib_dir.mkdir(exist_ok=True)

        # Create build directory, cleaning if necessary
        build_dir = mpiwrapper_dir / "build"
        self._ensure_clean_build_dir(build_dir, mpiwrapper_dir)

        # Handle macOS CMakeLists.txt patching
        self._patch_cmake_on_macos(mpiwrapper_dir)

        try:
            # Configure CMake
            cmake_args = [
                "cmake",
                "..",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={package_lib_dir.absolute()}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={package_lib_dir.absolute()}",
            ]
            subprocess.check_call(cmake_args, cwd=build_dir)

            # Build
            subprocess.check_call(["make"], cwd=build_dir)

            # Verify build
            target_lib = package_lib_dir / "libmpiwrapper.so"
            if not target_lib.exists():
                raise RuntimeError(f"Built library not found at {target_lib}")

            print(f"Successfully built MPIWrapper library at {target_lib}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error building MPIWrapper: {e}")

    def _patch_cmake_on_macos(self, mpiwrapper_dir: Path) -> None:
        """Patch CMakeLists.txt on macOS to disable two-level namespace check."""
        if sys.platform != "darwin":
            return

        cmake_file = mpiwrapper_dir / "CMakeLists.txt"
        if not cmake_file.exists():
            return

        with open(cmake_file, "r") as f:
            content = f.read()

        if "check_twolevel.sh" not in content:
            return

        # Comment out the two-level namespace check
        content = content.replace(
            "  add_custom_command(\n    TARGET mpiwrapper POST_BUILD\n    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_twolevel.sh ${CMAKE_CURRENT_BINARY_DIR}/libmpiwrapper.so",
            "  # add_custom_command(\n  #   TARGET mpiwrapper POST_BUILD\n  #   COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_twolevel.sh ${CMAKE_CURRENT_BINARY_DIR}/libmpiwrapper.so",
        )
        content = content.replace(
            '    COMMENT "Checking whether libmpiwrapper.so plugin uses a two-level namespace..."\n    VERBATIM\n    )',
            '  #   COMMENT "Checking whether libmpiwrapper.so plugin uses a two-level namespace..."\n  #   VERBATIM\n  #   )',
        )

        with open(cmake_file, "w") as f:
            f.write(content)

    def _ensure_clean_build_dir(self, build_dir: Path, mpiwrapper_dir: Path) -> None:
        """Ensure build directory is clean and doesn't have stale CMake cache."""
        cmake_cache = build_dir / "CMakeCache.txt"

        # If build directory exists and has a CMakeCache.txt, check if it's stale
        if build_dir.exists() and cmake_cache.exists():
            try:
                with open(cmake_cache, "r") as f:
                    cache_content = f.read()

                # Check if the cache references the current source directory
                current_source = str(mpiwrapper_dir.absolute())
                if current_source not in cache_content:
                    print(f"Detected stale CMake cache in {build_dir}, cleaning...")
                    shutil.rmtree(build_dir)

            except Exception as e:
                print(
                    f"Warning: Could not read CMake cache, cleaning build directory: {e}"
                )
                shutil.rmtree(build_dir)

        # Create build directory
        build_dir.mkdir(exist_ok=True)
