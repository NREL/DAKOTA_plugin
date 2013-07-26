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

/* This file will implement a simple boost based python binding for the dakota library
   Goal is to be able to simply say

   import dakota
   dakota.run()

   and have dakota do its thing
*/


#ifdef WINDOWS
#include "windows.h"
#endif

//#include "string.h"
//#include "mpi.h"


//#include <boost/python/module.hpp>
//#include <boost/python/def.hpp>
//#include <boost/python/tuple.hpp>
//#include <boost/python/numeric.hpp>
//#include <boost/python/extract.hpp>
//#include <boost/mpi.hpp>
//#include <numpy/arrayobject.h>


//namespace bp = boost::python;
//namespace bpn = boost::python::numeric;

#include "stdio.h"

/*  previous mac "bug" (turned out to be wrong combo of compilers being used).
///////////////////
// causes crash if:
// (1) we build with gcc 4.5, and (2) we import openmdao _before_ we import the library built here.
// something in dependencies already done with gcc 4.3 or lower causing crash. solution was to use gcc <= 4.3
#include <fstream>
/// file stream for tabulation of graphics data within compute_response 
std::ofstream tabularDataFStream;
////////////////////
*/

void test_fn()
{
  printf ("in test fn\n");
}

#include <boost/python.hpp>
//#include "/opt/local//Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy/core/include/numpy/ndarrayobject.h"
using namespace boost::python;
BOOST_PYTHON_MODULE(_testbinding)
{
  printf ("some sort of init happening!\n");

  //using namespace bpn;
  //  import_array();
  //array::set_module_and_type("numpy", "ndarray");
  //import_array();
  def("test_fn", test_fn, "test_fn");
}
