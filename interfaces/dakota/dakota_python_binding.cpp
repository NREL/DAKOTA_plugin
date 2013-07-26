// Copyright 2013 National Renewable Energy Laboratory (NREL)
// 
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
// 
//        http://www.apache.org/licenses/LICENSE-2.0
// 
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
// 
// ++==++==++==++==++==++==++==++==++==++==

/* dakota_python_interface.cpp

This file will implement a simple boost based python binding for the dakota library
   Goal is to be able to simply say

   import dakota
   dakota.run()

   and have dakota do its thing

   In actual fact we implement 3 ways:
   1) run_dakota(file) -- no args, just run it as if from the comnand line, with input file "file"
   2) run_dakota_data(file, data) -- "data" is any python object, it is passed back to your interface function
   3) run_dakota_mpi_data(file, comm, data) -- "comm" is an mpi_communicator over which the work is divided
   

   Windows build notes (SEE ADDENDUM BELOW):
   -------------------
   Building this interface on Windows (for a native Windows python, to be run from a DOS box) 
   was a project, so I'll make some notes here:
   In the end, success came from using the MinGW gcc compiler exclusively.  That is, I built DAKOTA from source with
   it (configure / make, see config.log in windows_build_stuff/), 
   and built the interface with it too.  For the latter, I used boost.build, i.e. "bjam".  See Jamroot.jam in
   this directory.

   False steps and why they don't work:  
      I first built the whole interface using cygwin.  This required
   building DAKOTA from source in cygwin, because the libraries supplied by Sandia use a different enough compiler that
   the name-mangling was a little off, and the link failed.  This interface worked, but only WITHIN cygwin.  This is 
   because it is linked against cygwin's python library.  Attempts to use the native python from cygwin resulted in
   either link errors or a module that would not load because it was missing dlls.  This route may be possible but I
   did not reach the end of it.

      Next I decided to use MSVC.  First I did this using bjam. bjam could make me a python module that worked with 
   the native python.  But it had a strange glitch r.e. the "regex" library that it kept building but then claimed to not
   be able to find.  So I tried building in the MSVC GUI, where I could specify exactly the libraries I wanted to link.
   This resulted in the realization that MSVC was never going to work unless I build all of DAKOTA with it too, i.e.
   massive link errors from compiling my interface with "cl.exe" and trying to link with gcc-build dakota libraries.

      Finally I learned that MinGW is not just "cygwin light".  It is actually a unix-like shell, including gcc,
   that builds native windows code.  However, mingw is not officially supported by boost.... and not anymore by DAKOTA... 

   ADDENDUM: Brian Adams of DAKOTA informs me that the only currently supported build systems for their Windows builds is
   cygwin.  Therefore, I highly recommend that anyone who inherits this code work within cygwin and figure out how to make that
   path work.  In particular, you will have to figure out how to make sure the _native_ python library is wired in, not
   the cygwin one.
*/

#include "string.h"
#ifndef WINDOWS
#include "mpi.h"
#else
#define MPI_Comm int
#define MPI_COMM_WORLD 0
#include "windows.h"
#endif


#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#ifndef WINDOWS
#include <boost/mpi.hpp>
#endif

#ifndef WINDOWS
#include <windows.h>
#endif

#include <numpy/arrayobject.h>


namespace bp = boost::python;
namespace bpn = boost::python::numeric;


// This function is in dakface.cpp, replaces those
int all_but_actual_main_mpi_data(int argc, char* argv[], MPI_Comm comm, void *data);
int all_but_actual_main(int argc, char* argv[]);

void run_dakota(char *infile)
{
  char app[10]; strcpy(app,"dakota");
  char *argv[2] = {app, infile};
  printf ("calling to DAKOTA all_but_actual_main\n");
  all_but_actual_main(2, argv);
}


#if 0
void run_dakota_mpi(char *infile, boost::mpi::communicator &_mpi)
{
  char app[10]; strcpy(app,"dakota");
  char *argv[] = {app, infile};
  MPI_Comm comm = _mpi;
  printf ("calling to DAKOTA all_but_actual_main_mpi\n");
  all_but_actual_main_mpi(2, argv, comm);
}
#endif

#ifdef WINDOWS
void run_dakota_mpi_data(char *infile, int &_mpi, bp::object data)
#else
void run_dakota_mpi_data(char *infile, boost::mpi::communicator &_mpi, bp::object data)
#endif
{
  char app[10]; strcpy(app,"dakota");
  char *argv[] = {app, infile};
  MPI_Comm comm = MPI_COMM_WORLD;

  if (_mpi) 
    comm = _mpi;
  void *tmp = NULL;
  if (data)
    tmp = &data;
  printf ("calling to DAKOTA all_but_actual_main_mpi_driver\n");
  int res = all_but_actual_main_mpi_data(2, argv, comm, tmp);
  //int res = all_but_actual_main_core(2, argv, &comm, NULL);
  printf ("made it out of dakface, but still in C++\n");
}

void run_dakota_data(char *infile,  bp::object data)
{
  char app[10]; strcpy(app,"dakota");
  char *argv[] = {app, infile};
  MPI_Comm comm = MPI_COMM_WORLD;

  void *tmp = NULL;
  if (data)
    tmp = &data;
  printf ("calling to DAKOTA all_but_actual_main_mpi_driver\n");
  int res = all_but_actual_main_mpi_data(2, argv, comm, tmp);
  //int res = all_but_actual_main_core(2, argv, &comm, NULL);
  printf ("made it out of dakface, but still in C++\n");
}

void test_fn()
{
  printf ("in test fn\n");
}

#include <boost/python.hpp>
//#include "/opt/local//Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy/core/include/numpy/ndarrayobject.h"
using namespace boost::python;
BOOST_PYTHON_MODULE(_dakota)
{
  using namespace bpn;
  import_array();
  array::set_module_and_type("numpy", "ndarray");
  //import_array();
  def("run_dakota", run_dakota, "run dakota");
  def("test_fn", test_fn, "test_fn");
  //  def("run_dakota_mpi", run_dakota_mpi, "run dakota mpi");
  // def("run_dakota_mpi_driver", run_dakota_mpi_driver, "run dakota mpi driver");
  def("run_dakota_mpi_data", run_dakota_mpi_data, "run dakota mpi data");
  def("run_dakota_data", run_dakota_data, "run dakota data");
}
