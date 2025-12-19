import os
import sys
import platform
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_config_var
from sysconfig import get_paths

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir, build_with_cuda):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.build_with_cuda = build_with_cuda

class Build(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        super().run()

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

            if ext.build_with_cuda:
                cmake_args += ['-DDIFFVG_CUDA=1']
            else:
                cmake_args += ['-DDIFFVG_CUDA=0']

            env = os.environ.copy()
            env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                                  self.distribution.get_version())
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

            # Rename the output file to match what python expects (e.g. diffvg.cpython-312-x86_64-linux-gnu.so)
            # CMake produces 'diffvg.so' on Linux because of PROPERTIES PREFIX "" in CMakeLists.txt
            if platform.system() != "Windows":
                built_path = os.path.join(extdir, 'diffvg.so')
                dest_path = self.get_ext_fullpath(ext.name)
                if os.path.exists(built_path) and built_path != dest_path:
                    print(f"Renaming {built_path} to {dest_path}")
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.rename(built_path, dest_path)
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

setup(
    packages=find_packages(include=['sketchkit*']) + ['pydiffvg'],
    package_dir={'pydiffvg': 'third_party/diffvg/pydiffvg'},
    ext_modules=[CMakeExtension('diffvg', 'third_party/diffvg', build_with_cuda)],
    cmdclass=dict(build_ext=Build),
    zip_safe=False,
    include_package_data=True,
    package_data={
        'sketchkit.sketch2model.methods.sens.spaghetti_deps': [
            '**/*.py',
        ],
    },
)
