# Copyright 2013 National Renewable Energy Laboratory (NREL)
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# 
# ++==++==++==++==++==++==++==++==++==++==
"""
Build pyDAKOTA Python 'egg' for cygwin, darwin, linux, or win32 platforms.
Assumes DAKOTA has been installed.  The egg will include all libraries
included in the DAKOTA installation.

The cygwin platform is built using gcc.  It requires a cygwin Python.
This has only been tested on the build machine.

The darwin platform is built using mpicxx.  This currently has problems when
trying to load the egg.  It's probably related to DAKOTA delivering i386 only
and the machine attempting to run is x86_64 => Python executable attempting
to load is likely running as x86_64.

The linux platform is built using mpicxx.  Some linker magic is used to avoid
having to set LD_LIBRARY_PATH on the system the egg is installed on.
Libraries not part of DAKOTA (boost, openmpi) are not included.
This has only been tested on the build machine.

The win32 platform is built using VisualStudio C++ and Intel Fortran.
This has been tested on a 'vanilla' (no DAKOTA pre-installed) Windows machine.
"""

import glob
import os.path
import shutil
import subprocess
import sys

from distutils.spawn import find_executable
from pkg_resources import get_build_platform
from setuptools import setup
from setuptools.extension import Extension

# Locate numpy include directory.
import numpy
numpy_include = os.path.dirname(numpy.__file__)+'/core/include'

# Execute DAKOTA to get version.
try:
    proc = subprocess.Popen(['dakota', '-v'], universal_newlines=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
except Exception, exc:
    print "Couldn't execute 'dakota -v':", exc
    sys.exit(1)

fields = stdout.split()
if len(fields) >= 3 and fields[0] == 'DAKOTA' and fields[1] == 'version':
    dakota_version = fields[2]
    wrapper_version = '1'
    egg_dir = 'pyDAKOTA-%s_%s-py%s-%s.egg' % (dakota_version, wrapper_version,
                                              sys.version[0:3], get_build_platform())
else:
    print "Can't parse version from DAKOTA output %r" % stdout
    print "    stderr output:", stderr
    sys.exit(1)

# Assuming standard prefix-based install.
dakota_install = os.path.dirname(
                     os.path.dirname(
                         find_executable('dakota')))
dakota_bin     = os.path.join(dakota_install, 'bin')
dakota_include = os.path.join(dakota_install, 'include')
dakota_lib     = os.path.join(dakota_install, 'lib')
if not os.path.exists(dakota_bin) or \
   not os.path.exists(dakota_include) or \
   not os.path.exists(dakota_lib):
    print "Can't find", dakota_bin, 'or', dakota_include, 'or', dakota_lib, ', bummer'
    sys.exit(1)

# Set to a list of any special compiler flags required.
CXX_FLAGS = []

# Set to a list of any special linker flags required.
LD_FLAGS = []

# Set to directory with 'boost' subdirectory (or None if found by default).
BOOST_INCDIR = None

# Set to directory with Boost libraries (or None if found by default).
BOOST_LIBDIR = None

# Set this for formatting library names like 'boost_regex' to library names for
# the linker.
BOOST_LIBFMT = '%s'
BOOST_PYFMT = None  # Used to handle case when only boost_python was built
                    # as shared library on Windows (temporary hack).

# Set to directory with LAPACK and BLAS libraries (or None if found by default).
LAPACK_LIBDIR = None

# Set to directory with Fortran libraries (or None if found by default).
FORTRAN_LIBDIR = None

# Set this to a list of extra libraries required beyond DAKOTA and BOOST.
EXTRA_LIBS = []

# Set this to a list of libraries to be included in the egg.
EGG_LIBS = []

# Set True to include MPI support.
NEED_MPI = False

# Set HAVE_ABORT_RETURNS True if DAKOTA has been updated to include the
# abort_returns flag.  This allows model exceptions to be passed back to the
# driver rather than causing the simulation to exit.
HAVE_ABORT_RETURNS = False


if sys.platform == 'cygwin':
    BOOST_LIBFMT = '%s-mt'
    EXTRA_LIBS = ['DGraphics', 'gfortran',
                  'SM', 'ICE', 'Xext', 'Xm', 'Xt', 'X11', 'Xpm', 'Xmu']
    EGG_LIBS = ['C:/cygwin/bin/cygboost_python-mt-1_50.dll',
                'C:/cygwin/bin/libpython2.7.dll']
    EGG_LIBS.extend(glob.glob(os.path.join(dakota_lib, '*.dll')))
    EGG_LIBS.extend(glob.glob(os.path.join(dakota_bin, '*.dll')))
    # Needed to avoid a problem with 'locale::facet::_S_create_c_locale'
    os.environ['LC_ALL'] = 'C'

elif sys.platform == 'win32':
    CXX_FLAGS = ['/EHsc']
    BOOST_INCDIR = r'C:\Users\setowns1\Downloads\boost_1_50_0'
    BOOST_LIBDIR = r'C:\Users\setowns1\Downloads\boost_1_50_0\stage\lib'
    BOOST_LIBFMT = 'lib%s-vc90-mt-1_50'
    BOOST_PYFMT = '%s-vc90-mt-1_50'  # Only built boost_python shared.
    LAPACK_LIBDIR = r'C:\Users\setowns1\Downloads\lapack-3.4.1\install\lib'
    FORTRAN_LIBDIR = r'C:\Program Files\Intel\Composer XE 2013\compiler\lib\ia32'
    EXTRA_LIBS = ['Dbghelp']
    EGG_LIBS = [
        r'C:\Users\setowns1\Downloads\boost_1_50_0\stage\lib\boost_python-vc90-mt-1_50.dll',
        r'C:\Program Files\Intel\Composer XE 2013\redist\ia32\compiler\libifcoremd.dll',
        r'C:\Program Files\Intel\Composer XE 2013\redist\ia32\compiler\libmmd.dll',
        r'C:\Program Files\Intel\Composer XE 2013\redist\ia32\compiler\svml_dispmd.dll',
        r'C:\Program Files\Microsoft Visual Studio 9.0\VC\redist\x86\Microsoft.VC90.CRT\Microsoft.VC90.CRT.manifest',
        r'C:\Program Files\Microsoft Visual Studio 9.0\VC\redist\x86\Microsoft.VC90.CRT\msvcm90.dll',
        r'C:\Program Files\Microsoft Visual Studio 9.0\VC\redist\x86\Microsoft.VC90.CRT\msvcp90.dll',
        r'C:\Program Files\Microsoft Visual Studio 9.0\VC\redist\x86\Microsoft.VC90.CRT\msvcr90.dll']
    HAVE_ABORT_RETURNS = True

elif sys.platform == 'darwin':
    # This builds an egg, but it doesn't load:
    #   ImportError: dlopen(/Users/setowns1/OpenMDAO-Framework/devenv/lib/python2.7/site-packages/pyDAKOTA-5.3.1_1-py2.7-macosx-10.6-intel.egg/pyDAKOTA.so, 2): Symbol not found: __ZN3MPI3Win4FreeEv
    #   Referenced from: /Users/setowns1/OpenMDAO-Framework/devenv/lib/python2.7/site-packages/pyDAKOTA-5.3.1_1-py2.7-macosx-10.6-intel.egg/pyDAKOTA.so
    #   Expected in: flat namespace
    # The symbol exists in the OpenMPI libraries in DYLD_LIBRARY_PATH.
    # Possibly something related to i386/x86_64 architecture builds.
    # (DAKOTA libraries are only i386)
    BOOST_INCDIR = '/Users/setowns1/include'
    BOOST_LIBDIR = '/Users/setowns1/boost_1_50_0/stage/lib'
    EXTRA_LIBS = ['DGraphics', 'gfortran.3', 'Xm.3']
    EGG_LIBS = glob.glob(os.path.join(dakota_lib, '*.dylib'))
    EGG_LIBS.extend(glob.glob(os.path.join(dakota_bin, '*.dylib')))
    NEED_MPI = True

else:
    # This LD_FLAGS stuff avoids having to set LD_LIBRARY_PATH to access
    # the other (not pyDAKOTA.so) shared libraries that are part of the egg.
    LD_FLAGS = ['-Wl,-z origin',
                '-Wl,-rpath=${ORIGIN}:${ORIGIN}/../'+egg_dir]
    EXTRA_LIBS = ['DGraphics', 'gfortran',
                  'SM', 'ICE', 'Xext', 'Xm', 'Xt', 'X11', 'Xpm', 'Xmu']
    EGG_LIBS = glob.glob(os.path.join(dakota_lib, '*.so'))
    EGG_LIBS.extend(glob.glob(os.path.join(dakota_bin, '*.so')))
    NEED_MPI = True
    HAVE_ABORT_RETURNS = True


sources = [
    'dakface.cpp', 'dakota_python_binding.cpp', 'python_interface.cpp',
]

include_dirs = [dakota_include, numpy_include]
if BOOST_INCDIR:
    include_dirs.append(BOOST_INCDIR)

# HAVE_AMPL *must* match install (at least for v5.3) for structure size match.
define_macros = [
    ('DAKOTA_PYTHON_NUMPY', None),
    ('HAVE_AMPL', None),
]
if HAVE_ABORT_RETURNS:
    define_macros.append(('HAVE_ABORT_RETURNS', None))

# Some DAKOTA distributions (i.e. cygwin) put libraries in 'bin'.
library_dirs = [dakota_lib, dakota_bin]
if BOOST_LIBDIR:
    library_dirs.append(BOOST_LIBDIR)
if LAPACK_LIBDIR:
    library_dirs.append(LAPACK_LIBDIR)
if FORTRAN_LIBDIR:
    library_dirs.append(FORTRAN_LIBDIR)

# Based on TPL LIBS displayed near end of cmake output.
# 'jega' repeated for non-rescanning linkers.
dakota_libs = [
    'dakota_src', 'dakota_src_fortran',
    'nidr', 'teuchos', 'pecos', 'pecos_src', 'lhs', 'mods', 'mod', 'dfftpack',
    'sparsegrid', 'surfpack', 'surfpack_fortran', 'colin', 'interfaces',
    'scolib', '3po', 'pebbl', 'utilib', 'tinyxml', 'conmin', 'dace',
    'analyzer', 'random', 'sampling', 'bose', 'fsudace', 'hopspack',
    'jega_fe', 'jega', 'moga', 'soga', 'jega', 'eutils', 'utilities', 'ncsuopt',
    'cport', 'optpp', 'psuade', 'amplsolver']

# Based on EXTRA TPL LIBS displayed near end of cmake output.
external_libs = [
    'boost_regex', 'boost_filesystem', 'boost_system', 'boost_signals',
    'boost_python', 'lapack', 'blas']

if NEED_MPI:
    define_macros.append(('DAKOTA_HAVE_MPI', None))
    external_libs.append('boost_mpi')
    os.environ['CC'] = 'mpicxx'  # Force compiler command.

# Munge boost library names as necessary.
if BOOST_LIBFMT:
    for i, name in enumerate(external_libs):
        if name.startswith('boost_'):
            if name == 'boost_python' and BOOST_PYFMT:
                external_libs[i] = BOOST_PYFMT % name
            else:
                external_libs[i] = BOOST_LIBFMT % name

libraries = dakota_libs + external_libs + EXTRA_LIBS

# List extra files to be included in the egg.
data_files = []
if EGG_LIBS:
    with open('MANIFEST.in', 'w') as manifest:
        for lib in EGG_LIBS:
            manifest.write('include %s\n' % os.path.basename(lib))
    data_files = [('', EGG_LIBS)]

pyDAKOTA = Extension(name='pyDAKOTA',
                     sources=sources,
                     include_dirs=include_dirs,
                     define_macros=define_macros,
                     extra_compile_args=CXX_FLAGS,
                     extra_link_args=LD_FLAGS,
                     library_dirs=library_dirs,
                     libraries=libraries,
                     language='c++')

setup(name='pyDAKOTA',
      version='%s-%s' % (dakota_version, wrapper_version),
      description='A Python wrapper for DAKOTA',
      py_modules=['dakota', 'test_dakota'],
      ext_modules=[pyDAKOTA],
      zip_safe=False,
      data_files=data_files)

