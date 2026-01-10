"""Setup script for Teddy C++ module compilation and installation."""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import os
from pathlib import Path

# Get the directory containing this setup.py
TEDDY_ROOT = Path(__file__).parent.resolve()


class TeddyBuildExt(build_ext):
    """Custom build extension to compile Teddy C++ module using CMake."""
    
    def run(self):
        """Build Teddy C++ module using CMake."""
        print("\n" + "="*60)
        print("Building Teddy C++ extension")
        print("="*60)
        
        build_dir = TEDDY_ROOT / "build"
        
        # Check if already built
        if build_dir.exists():
            so_files = list(build_dir.glob("teddy_sketch2model*.so"))
            if so_files:
                print(f"✓ Teddy module already built: {so_files[0].name}")
                print(f"  Location: {so_files[0]}")
                return
        
        # Build with CMake
        try:
            build_dir.mkdir(exist_ok=True)
            original_cwd = os.getcwd()
            
            os.chdir(str(build_dir))
            
            # Configure - use current Python interpreter
            print("\nConfiguring CMake...")
            print(f"  CMake source: {TEDDY_ROOT}")
            print(f"  Build directory: {build_dir}")
            print(f"  Python executable: {sys.executable}")
            print(f"  Python version: {sys.version}")
            
            # Pass Python executable to CMake
            cmake_args = [
                "cmake",
                "..",
                f"-DPython3_EXECUTABLE={sys.executable}"
            ]
            subprocess.check_call(cmake_args, cwd=str(build_dir))
            
            # Build
            print("\nBuilding...")
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", "Release"],
                cwd=str(build_dir)
            )
            
            print("\n✓ Teddy C++ module built successfully!")
            so_files = list(build_dir.glob("teddy_sketch2model*.so"))
            if so_files:
                print(f"  Module: {so_files[0].name}")
            print(f"  Location: {build_dir}")
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed to build Teddy C++ module: {e}")
            print("\nYou can build it manually using:")
            print(f"  cd {TEDDY_ROOT}")
            print("  mkdir build && cd build")
            print(f"  cmake .. -DPython3_EXECUTABLE={sys.executable}")
            print("  cmake --build . --config Release")
            sys.exit(1)
        except FileNotFoundError:
            print("\n✗ CMake not found. Please install CMake:")
            print("  Linux: sudo apt-get install cmake build-essential python3-dev")
            print("  macOS: brew install cmake")
            print("\nTeddy method will not be available until CMake is installed.")
            sys.exit(1)
        finally:
            os.chdir(original_cwd)


setup(
    name="teddy-sketch2model",
    version="0.1.0",
    description="Teddy/MonsterMash C++ module for sketch-to-3D conversion",
    author="SketchKit Contributors",
    url="https://github.com/CISLab-HKUST/SketchKit",
    py_modules=[],
    cmdclass={
        "build_ext": TeddyBuildExt,
    },
    zip_safe=False,
)

