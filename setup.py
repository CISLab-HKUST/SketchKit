import os
import sys
import platform
import subprocess
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_config_var
from sysconfig import get_paths

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir, build_with_cuda):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.build_with_cuda = build_with_cuda

class Build(build_ext):
    def run(self):
        # Build rtsc and trimesh2
        self.build_rtsc()

        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        super().run()

    def build_rtsc(self):
        # Detect platform
        system = platform.system()
        machine = platform.machine()
        
        target = None
        binary_name = 'nogui_contour_renderer'
        
        if system == 'Linux':
            if machine == 'x86_64':
                target = 'linux64'
            else:
                target = 'linux32'
        elif system == 'Darwin':
             target = 'darwin64'
        elif system == 'Windows':
             target = 'win32'
             binary_name += '.exe'
        
        if not target:
            print(f"Warning: Unknown platform {system} {machine}, skipping rtsc build.")
            return

        cwd = os.getcwd()
        third_party_dir = os.path.join(cwd, 'third_party')
        trimesh_dir = os.path.join(third_party_dir, 'trimesh2')
        rtsc_dir = os.path.join(third_party_dir, 'rtsc-1.6')
        
        # Check if directories exist
        if not os.path.exists(trimesh_dir) or not os.path.exists(rtsc_dir):
             print("Warning: third_party directories not found, skipping rtsc build.")
             return

        # Build trimesh2
        print(f"Building trimesh2 ({target})...")
        try:
            subprocess.check_call(['make', target], cwd=trimesh_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error building trimesh2: {e}")
            return

        # Build rtsc
        print(f"Building rtsc ({target})...")
        try:
            subprocess.check_call(['make', target], cwd=rtsc_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error building rtsc: {e}")
            return
        
        src_binary = os.path.join(rtsc_dir, binary_name)
        dest_dir = os.path.join(cwd, 'sketchkit', 'renderer')
        dest_binary = os.path.join(dest_dir, binary_name)
        
        if os.path.exists(src_binary):
            print(f"Moving {src_binary} to {dest_binary}")
            if os.path.exists(dest_binary):
                os.remove(dest_binary)
            shutil.copy2(src_binary, dest_binary)
            # Make executable
            st = os.stat(dest_binary)
            os.chmod(dest_binary, st.st_mode | 0o111)
        else:
            print(f"Warning: {src_binary} not found after build.")

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            info = get_paths()
            include_path = info['include']
            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                          '-DPYTHON_LIBRARY=' + get_config_var('LIBDIR'),
                          '-DPYTHON_INCLUDE_PATH=' + include_path,
                          '-DCMAKE_POLICY_VERSION_MINIMUM=3.5']

            cfg = 'Debug' if self.debug else 'Release'
            build_args = ['--config', cfg]

            if platform.system() == "Windows":
                cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                               '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
                if sys.maxsize > 2**32:
                    cmake_args += ['-A', 'x64']
                build_args += ['--', '/m']
            else:
                cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
                build_args += ['--', '-j8']

            # Extension-specific CMake arguments
            if ext.name == 'teddy_sketch2model':
                # Teddy-specific: pass Python executable
                cmake_args += ['-DPython3_EXECUTABLE=' + sys.executable]
            elif ext.name == 'diffvg':
                # Diffvg-specific: CUDA support
                if ext.build_with_cuda:
                    cmake_args += ['-DDIFFVG_CUDA=1']
                else:
                    cmake_args += ['-DDIFFVG_CUDA=0']

            env = os.environ.copy()
            env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                                  self.distribution.get_version())
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            
            # Use separate build directory for each extension
            ext_build_temp = os.path.join(self.build_temp, ext.name)
            if not os.path.exists(ext_build_temp):
                os.makedirs(ext_build_temp)
            
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=ext_build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=ext_build_temp)

            # Rename the output file to match what python expects
            if platform.system() != "Windows":
                if ext.name == 'diffvg':
                    # Diffvg produces 'diffvg.so' and needs to be renamed to include Python version info
                    built_path = os.path.join(extdir, 'diffvg.so')
                    dest_path = self.get_ext_fullpath(ext.name)
                    if os.path.exists(built_path) and built_path != dest_path:
                        print(f"Renaming {built_path} to {dest_path}")
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        os.rename(built_path, dest_path)
                # Teddy uses pybind11_add_module which automatically generates the correct filename format
                # (e.g., teddy_sketch2model.cpython-38-x86_64-linux-gnu.so), so no rename needed
        else:
            super().build_extension(ext)

# Check for torch and CUDA
build_with_cuda = False
try:
    import torch
    if torch.cuda.is_available():
        build_with_cuda = True
except ImportError:
    pass

# Override build_with_cuda with environment variable
if 'DIFFVG_CUDA' in os.environ:
    build_with_cuda = os.environ['DIFFVG_CUDA'] == '1'

# Prepare extension modules
ext_modules = []
if os.path.exists(os.path.join(PROJECT_ROOT, 'third_party', 'diffvg')):
    ext_modules.append(CMakeExtension('diffvg', 'third_party/diffvg', build_with_cuda))
if os.path.exists(os.path.join(PROJECT_ROOT, 'third_party', 'teddy')):
    ext_modules.append(CMakeExtension('teddy_sketch2model', 'third_party/teddy', False))

setup(
    packages=find_packages(include=['sketchkit*']) + ['pydiffvg'],
    package_dir={'pydiffvg': 'third_party/diffvg/pydiffvg'},
    ext_modules=ext_modules,
    package_data={'sketchkit': ['renderer/nogui_contour_renderer', 'renderer/nogui_contour_renderer.exe']},
    cmdclass=dict(build_ext=Build),
    zip_safe=False,
)