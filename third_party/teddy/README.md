# Teddy Sketch-to-3D Core Algorithm Implementation

Core algorithm implementation for sketch-to-3D model conversion based on MonsterMash/Teddy. This directory contains the core algorithm code, and dependency libraries need to be downloaded and configured by users.

## Directory Structure

- `src/` - Core algorithm source code
- `pybind11/` - Python bindings
- `eigen3-mod/` - Eigen modified headers (required)
- `CMakeLists.txt` - CMake configuration

## Dependencies

### Setup Environment

```bash
conda create -n teddy_check python=3.8
conda activate teddy_check
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-dev
cd /path/to/wsl_package
python -m pip install -r requirements.txt
```

### External Libraries

1. **Eigen3**: `sudo apt-get install libeigen3-dev` or download from https://eigen.tuxfamily.org/
2. **libigl**: `git clone https://github.com/libigl/libigl.git`
3. **MonsterMash third_party**: `git clone https://github.com/google/monster-mash.git` (use `monster-mash/third_party` directory)

## Build Steps

### Option 1: Using setup.py (Recommended)

```bash
cd third_party/teddy
python setup.py build_ext
```

This will automatically:
- Create the `build/` directory
- Configure CMake
- Compile the C++ module

### Option 2: Manual CMake Build

```bash
cd third_party/teddy
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

**Note**: Dependencies (Eigen3, libigl, etc.) are bundled in `third_party/teddy/third_party/` and will be automatically found during compilation.

### Verify

Compiled module `teddy_sketch2model.cpython-*.so` will be in `build/` directory.

## Usage

```python
from sketchkit.sketch2model import sketch_to_3d
output_path = sketch_to_3d("sketch.png", method='teddy', output_path='output.obj')
```
