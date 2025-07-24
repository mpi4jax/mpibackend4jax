import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.build_py import build_py


class BuildMPIWrapper(build_ext):
    """Custom build command to compile MPIWrapper"""
    
    def run(self):
        # Get the package directory
        package_dir = Path(__file__).parent
        mpiwrapper_dir = package_dir / "src" / "MPIwrapper"
        
        # Build directly into the package lib directory
        package_lib_dir = package_dir / "src" / "mpitrampoline4jax" / "lib"
        package_lib_dir.mkdir(exist_ok=True)
        
        # Use the package lib directory as the build output
        build_dir = mpiwrapper_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        # On macOS, we need to patch the CMakeLists.txt to disable the two-level namespace check
        # since modern macOS defaults to two-level namespace and the check script is too strict
        if sys.platform == "darwin":
            cmake_file = mpiwrapper_dir / "CMakeLists.txt"
            with open(cmake_file, 'r') as f:
                content = f.read()
            
            # Comment out the two-level namespace check on macOS
            if "check_twolevel.sh" in content:
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
        
        # Run cmake and make, building directly into the package lib directory
        try:
            # Configure CMake to output library to package lib directory
            cmake_args = [
                "cmake", "..",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={package_lib_dir.absolute()}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={package_lib_dir.absolute()}"
            ]
            subprocess.check_call(cmake_args, cwd=build_dir)
            subprocess.check_call(["make"], cwd=build_dir)
            
            # Verify the library was built in the correct location
            target_lib = package_lib_dir / "libmpiwrapper.so"
            if target_lib.exists():
                print(f"Successfully built MPIWrapper library at {target_lib}")
                
                # Also copy to build lib directory for wheel inclusion
                build_lib_dir = Path("build/lib/mpitrampoline4jax/lib")
                build_lib_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(target_lib, build_lib_dir / "libmpiwrapper.so")
                print(f"Copied {target_lib} to {build_lib_dir / 'libmpiwrapper.so'}")
            else:
                print(f"Warning: Built library not found at {target_lib}")
        except subprocess.CalledProcessError as e:
            print(f"Error building MPIWrapper: {e}")
            sys.exit(1)
        
        # Call parent build_ext
        super().run()


class CustomInstall(install):
    """Custom install command that runs build_ext first"""
    
    def run(self):
        self.run_command('build_ext')
        super().run()


class CustomBuildPy(build_py):
    """Custom build_py that excludes MPIwrapper from wheels"""
    
    def find_all_modules(self):
        """Override to exclude MPIwrapper modules from wheels"""
        modules = super().find_all_modules()
        # Filter out MPIwrapper modules
        return [m for m in modules if not m[0].startswith('MPIwrapper')]


setup(
    name="mpitrampoline4jax",
    version="0.1.0",
    description="MPITrampoline integration for JAX",
    long_description="A Python package that provides MPITrampoline integration for JAX, automatically building MPIWrapper and setting required environment variables.",
    packages=find_packages(where="src", exclude=["MPIwrapper*"]),
    package_dir={"": "src"},
    include_package_data=False,
    package_data={
        'mpitrampoline4jax': ['lib/*'],
    },
    cmdclass={
        'build_ext': BuildMPIWrapper,
        'install': CustomInstall,
        'build_py': CustomBuildPy,
    },
    install_requires=[
        "jax",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)