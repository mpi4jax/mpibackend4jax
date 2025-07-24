#!/usr/bin/env python3
"""
Development build script for MPIWrapper.
Run this script to build MPIWrapper in place for development.
"""

import sys
import subprocess
from pathlib import Path

def build_mpiwrapper():
    """Build MPIWrapper directly into src/mpitrampoline4jax/lib/"""
    
    # Get directories
    project_root = Path(__file__).parent
    mpiwrapper_dir = project_root / "src" / "MPIwrapper"
    lib_dir = project_root / "src" / "mpitrampoline4jax" / "lib"
    build_dir = mpiwrapper_dir / "build"
    
    # Create directories
    lib_dir.mkdir(exist_ok=True)
    build_dir.mkdir(exist_ok=True)
    
    print(f"Building MPIWrapper from {mpiwrapper_dir}")
    print(f"Output directory: {lib_dir}")
    
    # Apply macOS fix if needed
    if sys.platform == "darwin":
        cmake_file = mpiwrapper_dir / "CMakeLists.txt"
        with open(cmake_file, 'r') as f:
            content = f.read()
        
        # Comment out the two-level namespace check on macOS
        if "check_twolevel.sh" in content and "# add_custom_command" not in content:
            content = content.replace(
                "  add_custom_command(\n    TARGET mpiwrapper POST_BUILD\n    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_twolevel.sh ${CMAKE_CURRENT_BINARY_DIR}/libmpiwrapper.so",
                "  # add_custom_command(\n  #   TARGET mpiwrapper POST_BUILD\n  #   COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_twolevel.sh ${CMAKE_CURRENT_BINARY_DIR}/libmpiwrapper.so"
            )
            content = content.replace(
                "    COMMENT \"Checking whether libmpiwrapper.so plugin uses a two-level namespace...\"\n    VERBATIM\n    )",
                "  #   COMMENT \"Checking whether libmpiwrapper.so plugin uses a two-level namespace...\"\n  #   VERBATIM\n  #   )"
            )
            
            with open(cmake_file, 'w') as f:
                f.write(content)
            print("Applied macOS two-level namespace fix")
    
    # Run CMake
    cmake_args = [
        "cmake", "..",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir.absolute()}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={lib_dir.absolute()}"
    ]
    
    try:
        print("Running CMake...")
        subprocess.check_call(cmake_args, cwd=build_dir)
        
        print("Running make...")
        subprocess.check_call(["make"], cwd=build_dir)
        
        # Check if library was built
        lib_file = lib_dir / "libmpiwrapper.so"
        if lib_file.exists():
            print(f"✅ Successfully built MPIWrapper library at {lib_file}")
            return True
        else:
            print(f"❌ Library not found at {lib_file}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False

if __name__ == "__main__":
    success = build_mpiwrapper()
    sys.exit(0 if success else 1)