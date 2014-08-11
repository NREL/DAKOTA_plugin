pyDAKOTA: a Python wrapper for DAKOTA
-------------------------------------

This is a generic Python wrapper for DAKOTA originally written by
Peter Graf, National Renewable Energy Lab, 2012. peter.graf@nrel.gov.
That code combined both the generic wrapper and an OpenMDAO 'driver'
which used the wrapper. For maintenance reasons, the code has been split
into this generic portion usable by any Python program, and a separate
OpenMDAO driver plugin.

The original code is at https://github.com/NREL/DAKOTA_plugin.
The file dakface.pdf provides some background on how the original code
was structured, and is generally valid with this updated version.

The OpenMDAO driver using this code is at
https://github.com/OpenMDAO-Plugins/dakota-driver.


This code provides:

1. An interface to DAKOTA, in "library mode", that supports passing to DAKOTA
argc/argv for the command-line, an optional MPI communicator, and a pointer
to a Python exception object. This is still in C++.

2. A Python wrapper for this interface, so, in Python, you can say
"import dakota", then "dakota.DakotaBase().run_dakota(mpi_comm=comm)".
"comm" will be used as the MPI communicator for DAKOTA, and
DakotaBase.dakota_callback() will be called by DAKOTA for function evaluations.

The deliverable is a Python 'egg'. If your environment is properly configured,
you can use this to build the egg:

    python setup.py bdist_egg -d .

To install the egg (easy_install is from setuptools):

    easy_install pyDAKOTA-6.0_1-py2.7-linux-x86_64.egg

To run a trivial test:

    python -m test_dakota

This has been tested on Linux and Windows. Cygwin has also been sucessfully
built in the past.


Requirements
------------

DAKOTA 6.0+ (svn trunk >= 2707).

Like the normal DAKOTA installation, the Linux egg does not provide various
dynamic libraries expected to come from the system.  For a 'vanilla' Ubuntu
test installation the following libraries were not included (specific versions
may vary with egg build):

    Boost
        libboost_filesystem.so.1.49.0
        libboost_mpi.so.1.49.0
        libboost_python.so.1.49.0
        libboost_regex.so.1.49.0
        libboost_serialization.so.1.49.0
        libboost_signals.so.1.49.0
        libboost_system.so.1.49.0

    OpenMPI
        libmpi.so.0
        libmpi_cxx.so.0
        libopen_pal.so.0
        libopen_rte.so.0

    Unicode:
        libicudata.so.48
        libicui8n.so.48
        libicuuc.so.48

    X:
        libXm.so.4
        libXp.so.6

Also like the normal DAKOTA installation, the Windows egg includes everything
necessary for a 'vanilla' Windows installation.


License
-------
This software is licensed under the Apache 2.0 license.
See "Apache2.0License.txt" in this directory.


C++ source code:
----------------
dakface.cpp:  This is the library entry point.  It runs DAKOTA in 'sandwich'
mode where the caller provides input and DAKOTA calls-back to perform function
evaluations.

dakota_python_binding.cpp:  This is the boost wrapper that exposes the
functions in dakface.cpp to python.

