NREL DAKOTA Driver for openMDAO
==============================

Feb, 2013, Peter Graf

This is a _very_ brief description of the materials for this "V-0.0.0.0.1" DakotaDriver.  
Please contact me (peter.graf@nrel.gov) for help finishing it off.

C++ source code:
--------------
dakface.h: header for the C++ materials.

dakface.cpp:  This is the library entry point.  Following the instructions at http://dakota.sandia.gov/docs/dakota/5.2/html-dev/DakLibrary.html, it detects if the user has chosen the "NRELpython" interface, and installs one if so.

dakota_python_binding.cpp:  This is the boost wrapper that exposes the functions in dakface.cpp to python.  There are some brief
Windows build notes in a comment in this file.

python_interface.cpp:  This is where the subclass of DAKOTA's DirectApplicInterface is implemented.  I had to copy/paste a lot
of code from DAKOTA because the functions I needed were declared "private".  However, Brian Adams at DAKOTA has informed me
that in the DAKOTA 5.3 release, there is a separate python interface subclass of DirectApplicInterface, and that the relevant
functions are now protected (so your subclass can use them).  So the new version of this file will be much shorter.  The
most important change from the regular python interface is that we  carry this void * around (in the DAKOTADriver openMDAO
object, the void * is the actual driver).  Starting on line 198 is where we bundle that up and send it to the python callback.

tdakota_main.cpp:  This is main() so that you can run a standalone Dakota that knows about our python interface (ie it 
calls dakface).  I used it mainly for testing.

test_python_binding.cpp: skeletal boost wrapper, for testing.

Mac Makefile
-----------
Makefile:  Build on the Mac was by a hacked up makefile, here it is.

Boost build files (Windows):
--------------------------
	Building on Windows was a royal pain.  Your best bet will be to stick to cygwin.  But you have to figure out how to 
get the boost python libraries build correctly, which involves making sure you are picking up the _native_ Windows python
libraries, not the ones that are part of cygwin.  I _did not_ do it this way, but ended up succeeding by using MinGW.  Brian
Adams tells me that cygwin is the DAKOTA/Windows build combo of choice for the time being, so I'd suggest figuring out
how to make it work.

Jamroot.jam: used to build boost libraries
user-config.jam:  tells boost what python to use
boost-build.jam:  tells boost where it's build system is.


DAKOTADriver openMDAO Driver:
---------------------------
__init__.py: the usual requirement to be in a package, no code here.
dakota_driver.py:  The openMDAO Driver.  This should be familiar to you!

misc:
----
make_openmdao_path.py: For running DAKOTA from command line, and having it call back to openMDAO stuff.  Needs to 
know the paths. Not necessary for verion here, which does not use system calls.


