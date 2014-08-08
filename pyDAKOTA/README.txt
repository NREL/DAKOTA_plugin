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

1. An interface to DAKOTA, in "library mode", that allows passing an MPI
communicator and a "void *" object to DAKOTA. This is still in C++.

2. A python wrapper for this interface, so, in python, you can say
"import dakota", then "dakota.run_dakota(infile, object, comm)".
"comm" will be used as the MPI communicator for DAKOTA, and "object" will be
passed _back_ to the python routine specified in your dakota input file.

The deliverable is a Python 'egg'. If your environment is properly configured,
you can use this to build the egg:

    python setup.py bdist_egg -d .

To install the egg (easy_install is from setuptools):

    easy_install pyDAKOTA-5.3_1-py2.7-linux-x86_64.egg

To run a trivial test:

    python -m test_dakota

This has been tested on Linux and Windows. Cygwin has also been sucessfully
built in the past.


License
-------
This software is licensed under the Apache 2.0 license.
See "Apache2.0License.txt" in this directory.


C++ source code:
----------------
dakface.cpp:  This is the library entry point.  Following the instructions at
http://dakota.sandia.gov/docs/dakota/5.2/html-dev/DakLibrary.html,
it detects if the user has chosen the "NRELpython" interface, and installs one
if so.

dakota_python_binding.cpp:  This is the boost wrapper that exposes the
functions in dakface.cpp to python.

python_interface.cpp:  This is where the subclass of DAKOTA's
DirectApplicInterface is implemented. This is basically identical to
PythonInterface, except:

1. It passes a user data pointer to the invoked Python function.
2. It can let Python exceptions propagate back to a Python caller of
   DAKOTA if other parts of DAKOTA have been updated to support this.
3. The constructor handles Python interpreter initialization better, in that if
   the intepreter is already running, we don't try to initialize it again.
   The interpreter would already be running if DAKOTA was invoked by a
   Python function (such as this wrapper).

This last point is why we can't simply inherit from PythonInterface.
Sadly, we have to duplicate it.

