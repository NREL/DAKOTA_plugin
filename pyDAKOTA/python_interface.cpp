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
/* python_interface.cpp

   This file implements an application interface for DAKOTA that is very much
   like the python-enabled direct application interface (DirectApplicInterface).
   The difference here is that we allow two more fields to be attached to the
   interface: 1) an MPI communicator, 2) a generic void *.  The void * is
   actually a python object, which can be used to do whatever you want, back on
   the python side, inside the function called by the interface.  The python
   function called by this interface is still specified as <module>:<function>
   in the input file to DAKOTA, unchanged from the Sandia python interface.
*/

// Must be before system includes according to Python docs.
//#include <Python.h>
// Replaces Python.h according to boost_python docs.
#include <boost/python/detail/wrap_python.hpp>

#include "python_interface.hpp"
#include "DataMethod.hpp"

#ifdef DAKOTA_PYTHON_NUMPY
#include <numpy/arrayobject.h>
#endif

#include <boost/python/def.hpp>
namespace bp = boost::python;

namespace Dakota {

/// The main point: a python interface that passes a python object back to the interface function
NRELPythonInterface::
NRELPythonInterface(const ProblemDescDB& problem_db, void* pData)
  : DirectApplicInterface(problem_db), _pUserData(pData), _myPython(false)
{
  if (!Py_IsInitialized()) {
    Py_Initialize();
    _myPython = true;
    if (Py_IsInitialized()) {
      if (outputLevel >= NORMAL_OUTPUT)
        Cout << "Python interpreter initialized for direct function evaluation."
  	     << std::endl;
    }
    else {
      Cerr << "Error: Could not initialize Python for direct function "
           << "evaluation." << std::endl;
      abort_handler(-1);
    }
  }
#ifdef DAKOTA_PYTHON_NUMPY
  import_array();
  //      userNumpyFlag = problem_db.get_bool("python_numpy");
  userNumpyFlag = true;
#else
  //      if (problem_db.get_bool("python_numpy")) {
  Cout << "Warning: Python numpy not available, ignoring user request."
       << std::endl;
  userNumpyFlag = false;
  //}
#endif
}


NRELPythonInterface::~NRELPythonInterface() {
  if (Py_IsInitialized() && _myPython) {
    Py_Finalize();
    if (outputLevel >= NORMAL_OUTPUT)
      Cout << "Python interpreter terminated." << std::endl;
  }
}


/// Python specialization of derived analysis components
int NRELPythonInterface::derived_map_ac(const String& ac_name)
{
#ifdef MPI_DEBUG
    Cout << "analysis server " << analysisServerId << " invoking " << ac_name
         << " within PythonInterface." << std::endl;
#endif // MPI_DEBUG

  int fail_code = python_run(ac_name);

  // Failure capturing
  if (fail_code)
    throw fail_code;

  return 0;
}


int NRELPythonInterface::python_run(const String& ac_name)
{
  // probably need to convert all of the following with SWIG or Boost!!
  // (there is minimal error checking for now)
  // need to cleanup ref counts on Python objects
  int fail_code = 0;

  // probably want to load the modules and functions at construction time, incl.
  // validation and store the objects for later, but need to resolve use of
  // analysisDriverIndex

  // must use empty tuple here to pass to function taking only kwargs
  PyObject *pArgs = PyTuple_New(0);
  PyObject *pDict = PyDict_New();

  // convert DAKOTA data types to Python objects (lists and/or numpy arrays)
  PyObject *cv, *cv_labels, *div, *div_labels, *drv, *drv_labels,
    *av, *av_labels, *asv, *dvv;
  python_convert(xC, &cv);
  python_convert(xCLabels, &cv_labels);
  python_convert_int(xDI, xDI.length(), &div);
  python_convert(xDILabels, &div_labels);
  python_convert(xDR, &drv);
  python_convert(xDRLabels, &drv_labels);
  python_convert(xC, xDI, xDR, &av);
  python_convert(xCLabels, xDILabels, xDRLabels, &av_labels);
  python_convert_int(directFnASV, directFnASV.size(), &asv);
  python_convert_int(directFnDVV, directFnDVV.size(), &dvv);
  // TO DO: analysis components

  // assemble everything into a dictionary to pass to user function
  // this should eat references to the objects declared above
  PyDict_SetItem(pDict, PyString_FromString("variables"), 
		 PyInt_FromLong((long) numVars));
  PyDict_SetItem(pDict, PyString_FromString("functions"), 
		 PyInt_FromLong((long) numFns)); 
  PyDict_SetItem(pDict, PyString_FromString("cv"), cv);
  PyDict_SetItem(pDict, PyString_FromString("cv_labels"), cv_labels);
  PyDict_SetItem(pDict, PyString_FromString("div"), div);
  PyDict_SetItem(pDict, PyString_FromString("div_labels"), div_labels);
  PyDict_SetItem(pDict, PyString_FromString("drv"), drv);
  PyDict_SetItem(pDict, PyString_FromString("drv_labels"), drv_labels);
  PyDict_SetItem(pDict, PyString_FromString("av"), av);
  PyDict_SetItem(pDict, PyString_FromString("av_labels"), av_labels);
  PyDict_SetItem(pDict, PyString_FromString("asv"), asv);
  PyDict_SetItem(pDict, PyString_FromString("dvv"), dvv);
  PyDict_SetItem(pDict, PyString_FromString("currEvalId"), 
		 PyInt_FromLong((long) currEvalId));

  // pass optional user data only if its been set
  if (_pUserData != NULL) {
    bp::object* tmp = (bp::object*)_pUserData;
    PyDict_SetItem(pDict, PyString_FromString("user_data"), tmp->ptr());
  }

#if 0
  // The active analysis_driver is passed in ac_name (in form
  // module:function); could make module optional.  We pass any
  // analysis components as string arguments to the Python function.
  size_t pos = ac_name.find(":");
  std::string module_name = ac_name.substr(0,pos);
  std::string function_name = ac_name.substr(pos+1);
#else
  // for now we presume a single analysis component containing module:function
  const std::string& an_comp = analysisComponents[analysisDriverIndex][0];
  size_t pos = an_comp.find(":");
  std::string module_name = an_comp.substr(0,pos);
  std::string function_name = an_comp.substr(pos+1);
#endif
  if (module_name.size() == 0 || function_name.size() == 0) {
    Cerr << "\nError: invalid Python analysis_driver '" << ac_name
	 << "'\n       Should have form 'module:function'." << std::endl;
    Py_DECREF(pDict);
    Py_DECREF(pArgs);    
    abort_handler(-1);
  }

  // import the module and function and test for callable
  PyObject *pModule = PyImport_Import(PyString_FromString(module_name.c_str()));
  if (pModule == NULL) {
    Cerr << "Error (NRELPythonInterface): Failure importing module '" 
	 << module_name  << "'.\n                         Consider setting "
	 << "PYTHONPATH." << std::endl;
    Py_DECREF(pDict);
    Py_DECREF(pArgs);    
    abort_handler(-1);
  }

  // Microsoft compiler chokes on this:
  //  char fn[function_name.size()+1];
  char *fn = new char[function_name.size()+1];
  strcpy(fn, function_name.c_str());
  PyObject *pFunc = PyObject_GetAttrString(pModule, fn);
  delete fn;
  if (!pFunc || !PyCallable_Check(pFunc)) {
    Cerr << "Error (NRELPythonInterface): Function '" << function_name  
	 << "' not found or not callable" << std::endl;
    Py_DECREF(pDict);
    Py_DECREF(pArgs);    
    Py_DECREF(pModule);
    abort_handler(-1);
  }

  // perform analysis
  if (outputLevel > NORMAL_OUTPUT)
    Cout << "Info (NRELPythonInterface): Calling function " << function_name 
	 << " in module " << module_name << "." << std::endl;
  PyErr_Clear();
  PyObject *retVal = PyObject_Call(pFunc, pArgs, pDict);

  Py_DECREF(pDict);
  Py_DECREF(pArgs);    
  Py_DECREF(pModule);
  Py_DECREF(pFunc);

  if (!retVal) {
    if (PyErr_Occurred()) {
#ifdef HAVE_ABORT_RETURNS
      if (!abort_returns)  // If configured to return, caller must report.
        PyErr_Print();
#endif
      fail_code = 1;
    }
    else {
      Cerr << "Error (NRELPythonInterface): Unknown error evaluating python "
	   << "function." << std::endl;
      fail_code = -1;
    }
    throw (fail_code);
  }

  // process the return data

  bool fn_flag = false;
  for (size_t i=0; i<numFns; ++i)
    if (directFnASV[i] & 1) {
      fn_flag = true;
      break;
    }

  // process return type as dictionary, else assume list of functions only
  if (PyDict_Check(retVal)) {
    // or the user may return a dictionary containing entires fns, fnGrads,
    // fnHessians, fnLabels, failure (int)
    // fnGrads, e.g. is a list of lists of doubles
    // this is where Boost or SWIG could really help
    // making a lot of assumptions on types being returned
    PyObject *obj;
    if (fn_flag) {
      if ( !(obj = PyDict_GetItemString(retVal, "fns")) ) {
	Cerr << "Python dictionary must contain list 'fns'" << std::endl;
	Py_DECREF(retVal);
	abort_handler(-1);
      }
      if (!python_convert(obj, fnVals, numFns)) {
	Py_DECREF(retVal);
	abort_handler(-1);
      }
    }
    if (gradFlag) {
      if ( !(obj = PyDict_GetItemString(retVal, "fnGrads")) ) {
	Cerr << "Python dictionary must contain list 'fnGrads'" << std::endl;
	Py_DECREF(retVal);
	abort_handler(-1);
      }
      if (!python_convert(obj, fnGrads)) {
	Py_DECREF(retVal);
	abort_handler(-1);
      }
    }
    if (hessFlag) {
      if ( !(obj = PyDict_GetItemString(retVal, "fnHessians")) ) {
	Cerr << "Python dictionary must contain list 'fnHessians'" << std::endl;
	Py_DECREF(retVal);
	abort_handler(-1);
      }
      if (!python_convert(obj, fnHessians)){
	Py_DECREF(retVal);
	abort_handler(-1);
      }
    }
    // optional returns
    if (obj = PyDict_GetItemString(retVal, "failure"))
      fail_code = PyInt_AsLong(obj);

    if (obj = PyDict_GetItemString(retVal, "fnLabels")) {
      if (!PyList_Check(obj) || PyList_Size(obj) != numFns) {
	Cerr << "'fnLabels' must be list of length numFns." << std::endl;
	Py_DECREF(retVal);
	abort_handler(-1);
      }
      for (size_t i=0; i<numFns; ++i)
	fnLabels[i] = PyString_AsString(PyList_GetItem(obj, i));
    }
  }
  else {
    // asssume list/numpy array containing only functions
    if (fn_flag)
      python_convert(retVal, fnVals, numFns);
  }
  Py_DECREF(retVal);

  return(fail_code);
}


/** convert all integer array types including IntVector, ShortArray,
    and SizetArray to Python list of ints or numpy array of ints */
template<class ArrayT, class Size>
bool NRELPythonInterface::
python_convert_int(const ArrayT& src, Size sz, PyObject** dst)
{
#ifdef DAKOTA_PYTHON_NUMPY
  if (userNumpyFlag) {
    npy_intp dims[1];
    dims[0] = sz;
    if (!(*dst = PyArray_SimpleNew(1, dims, PyArray_INT))) {
      Cerr << "Error creating Python numpy array." << std::endl;
      return(false);
    }
    PyArrayObject *pao = (PyArrayObject *) *dst;
    for (Size i=0; i<sz; ++i)
      *(int *)(pao->data + i*(pao->strides[0])) = (int) src[i];
  }
  else
#endif
  {
    if (!(*dst = PyList_New(sz))) {
      Cerr << "Error creating Python list." << std::endl;
      return(false);
    }
    for (Size i=0; i<sz; ++i)
      PyList_SetItem(*dst, i, PyInt_FromLong((long) src[i]));
  }
  return(true);
}


// convert RealVector to list of floats or numpy array of doubles
bool NRELPythonInterface::
python_convert(const RealVector& src, PyObject** dst)
{
  int sz = src.length();
#ifdef DAKOTA_PYTHON_NUMPY
  if (userNumpyFlag) {
    npy_intp dims[1];
    dims[0] = sz;
    if (!(*dst = PyArray_SimpleNew(1, dims, PyArray_DOUBLE))) {
      Cerr << "Error creating Python numpy array." << std::endl;
      return(false);
    }
    PyArrayObject *pao = (PyArrayObject *) *dst;
    for (int i=0; i<sz; ++i)
      *(double *)(pao->data + i*(pao->strides[0])) = src[i];
  }
  else
#endif
  {
    if (!(*dst = PyList_New(sz))) {
      Cerr << "Error creating Python list." << std::endl;
      return(false);
    }
    for (int i=0; i<sz; ++i)
      PyList_SetItem(*dst, i, PyFloat_FromDouble(src[i]));
  }
  return(true);
}


// helper for converting xC, xDI, and xDR to single Python array of all variables
bool NRELPythonInterface::
python_convert(const RealVector& c_src, const IntVector& di_src,
	       const RealVector& dr_src, PyObject** dst)
{
  int c_sz = c_src.length();
  int di_sz = di_src.length();
  int dr_sz = dr_src.length();
#ifdef DAKOTA_PYTHON_NUMPY
  if (userNumpyFlag) {
    npy_intp dims[1];
    dims[0] = c_sz + di_sz + dr_sz;
    if (!(*dst = PyArray_SimpleNew(1, dims, PyArray_DOUBLE))) {
      Cerr << "Error creating Python numpy array." << std::endl;
      return(false);
    }
    PyArrayObject *pao = (PyArrayObject *) *dst;
    for (int i=0; i<c_sz; ++i)
      *(double *)(pao->data + i*(pao->strides[0])) = c_src[i];
    for (int i=0; i<di_sz; ++i)
      *(double *)(pao->data + (c_sz+i)*(pao->strides[0])) = (double) di_src[i];
    for (int i=0; i<dr_sz; ++i)
      *(double *)(pao->data + (c_sz+di_sz+i)*(pao->strides[0])) = dr_src[i];
  }
  else
#endif
  {
    if (!(*dst = PyList_New(c_sz + di_sz + dr_sz))) {
      Cerr << "Error creating Python list." << std::endl;
      return(false);
    }
    for (int i=0; i<c_sz; ++i)
      PyList_SetItem(*dst, i, PyFloat_FromDouble(c_src[i]));
    for (int i=0; i<di_sz; ++i)
      PyList_SetItem(*dst, c_sz + i, PyInt_FromLong((long) di_src[i]));
    for (int i=0; i<dr_sz; ++i)
      PyList_SetItem(*dst, c_sz + di_sz + i, 
		     PyFloat_FromDouble(dr_src[i]));
  }
  return(true);
}


// convert StringArray to list of strings
bool NRELPythonInterface::
python_convert(const StringMultiArray& src, PyObject** dst)
{
  int sz = src.size();
  if (!(*dst = PyList_New(sz))) {
      Cerr << "Error creating Python list." << std::endl;
      return(false);
  }
  for (int i=0; i<sz; ++i)
    PyList_SetItem(*dst, i, PyString_FromString(src[i]));

  return(true);
}


// convert continuous and discrete label strings to single list
bool NRELPythonInterface::
python_convert(const StringMultiArray& c_src, const StringMultiArray& di_src,
	       const StringMultiArray& dr_src, PyObject** dst)
{
  int c_sz = c_src.size();
  int di_sz = di_src.size();
  int dr_sz = dr_src.size();
  if (!(*dst = PyList_New(c_sz + di_sz + dr_sz))) {
    Cerr << "Error creating Python list." << std::endl;
    return(false);
  }
  for (int i=0; i<c_sz; ++i)
    PyList_SetItem(*dst, i, PyString_FromString(c_src[i]));
  for (int i=0; i<di_sz; ++i)
    PyList_SetItem(*dst, c_sz+i, PyString_FromString(di_src[i]));
  for (int i=0; i<dr_sz; ++i)
    PyList_SetItem(*dst, c_sz+di_sz+i, PyString_FromString(dr_src[i]));

  return(true);
}


// Accepts python list or numpy array, DAKOTA RealVector,
// expected dimension. Returns false if conversion failed.
bool NRELPythonInterface::
python_convert(PyObject *pyv, RealVector& rv, const int& dim)
{
#ifdef DAKOTA_PYTHON_NUMPY
  // could automatically detect return type instead of throwing error
  if (userNumpyFlag) {
    if (!PyArray_Check(pyv) || PyArray_NDIM(pyv) != 1 || 
	PyArray_DIM(pyv,0) != dim) {
      Cerr << "Python numpy array not 1D of size " << dim << "." << std::endl;
      return(false);
    }
    PyArrayObject *pao = (PyArrayObject *) pyv;
    for (int i=0; i<dim; ++i)
      rv[i] = *(double *)(pao->data + i*(pao->strides[0]));
  }
  else
#endif
  {
    PyObject *val;
    if (!PyList_Check(pyv) || PyList_Size(pyv) != dim) {
      Cerr << "Python vector must have length " << dim << "." << std::endl;
      return(false);
    }
    for (int i=0; i<dim; ++i) {
      val = PyList_GetItem(pyv, i);
      if (PyFloat_Check(val))
	rv[i] = PyFloat_AsDouble(val);
      else if (PyInt_Check(val))
	rv[i] = (double) PyInt_AsLong(val);
      else {
	Cerr << "Unsupported Python data type converting vector." << std::endl;
	Py_DECREF(val);
	return(false);
      }
    }
  }
  return(true);
}

// Accepts python list or numpy array, pointer to double, e.g., view
// of a Teuchose::SerialDenseVector, expected dimension.  Returns
// false if conversion failed.
bool NRELPythonInterface::
python_convert(PyObject *pyv, double *rv, const int& dim)
{
#ifdef DAKOTA_PYTHON_NUMPY
  // could automatically detect return type instead of throwing error
  if (userNumpyFlag) {
    if (!PyArray_Check(pyv) || PyArray_NDIM(pyv) != 1 || 
	PyArray_DIM(pyv,0) != dim) {
      Cerr << "Python numpy array not 1D of size " << dim << "." << std::endl;
      return(false);
    }
    PyArrayObject *pao = (PyArrayObject *) pyv;
    for (int i=0; i<dim; ++i)
      rv[i] = *(double *)(pao->data + i*(pao->strides[0]));
  }
  else
#endif
  {
    PyObject *val;
    if (!PyList_Check(pyv) || PyList_Size(pyv) != dim) {
      Cerr << "Python vector must have length " << dim << "." << std::endl;
      return(false);
    }
    for (int i=0; i<dim; ++i) {
      val = PyList_GetItem(pyv, i);
      if (PyFloat_Check(val))
	rv[i] = PyFloat_AsDouble(val);
      else if (PyInt_Check(val))
	rv[i] = (double) PyInt_AsLong(val);
      else {
	Cerr << "Unsupported Python data type converting vector." << std::endl;
	Py_DECREF(val);
	return(false);
      }
    }
  }
  return(true);
}

// assume we're converting numFns x numDerivVars to numDerivVars x
// numFns (gradients) returns false if conversion failed
bool NRELPythonInterface::python_convert(PyObject *pym, RealMatrix &rm)
{
#ifdef DAKOTA_PYTHON_NUMPY
  if (userNumpyFlag) {
    if (!PyArray_Check(pym) || PyArray_NDIM(pym) != 2 || 
	PyArray_DIM(pym,0) != numFns  ||  PyArray_DIM(pym,1) != numDerivVars) {
      Cerr << "Python numpy array not 2D of size " << numFns << "x"
	   << numDerivVars << "." << std::endl;
      return(false);
    }
    PyArrayObject *pao = (PyArrayObject *) pym;
    for (size_t i=0; i<numFns; ++i)
      for (size_t j=0; j<numDerivVars; ++j)
	rm(j,i) = *(double *)(pao->data + i*(pao->strides[0]) + 
			      j*(pao->strides[1]));
  }
  else
#endif
  {
    PyObject *val;
    if (!PyList_Check(pym) || PyList_Size(pym) != numFns) {
      Cerr << "Python matrix must have " << numFns << "rows." << std::endl;
      return(false);
    }
    for (size_t i=0; i<numFns; ++i) {
      val = PyList_GetItem(pym, i);
      if (PyList_Check(val)) {
	// use the helper to convert this column of the gradients
	if (!python_convert(val, rm[i], numDerivVars))
	  return(false);
      }
      else {
	Cerr << "Each row of Python matrix must be a list." << std::endl;
	Py_DECREF(val);
	return(false);
      }
    }
  }
  return(true);
}

// assume numDerivVars x numDerivVars as helper in Hessian conversion
// and lower triangular storage in Hessians
// returns false if conversion failed
bool NRELPythonInterface::python_convert(PyObject *pym, 
					   RealSymMatrix &rm)
{
  // for now, the numpy case isn't called (since handled in calling
  // Hessian array convert)
#ifdef DAKOTA_PYTHON_NUMPY
  if (userNumpyFlag) {
    if (!PyArray_Check(pym) || PyArray_NDIM(pym) != 2 || 
	PyArray_DIM(pym,0) != numDerivVars  ||
	PyArray_DIM(pym,1) != numDerivVars) {
      Cerr << "Python numpy array not 2D of size " << numDerivVars << "x" 
	   << numDerivVars << "." << std::endl;
      return(false);
    }
    PyArrayObject *pao = (PyArrayObject *) pym;
    for (size_t i=0; i<numDerivVars; ++i)
      for (size_t j=0; j<=i; ++j)
	rm(i,j) = *(double *)(pao->data + i*(pao->strides[0]) + 
			       j*(pao->strides[1]));
  }
  else
#endif
  {
    if (!PyList_Check(pym) || PyList_Size(pym) != numDerivVars) {
      Cerr << "Python matrix must have " << numDerivVars << "rows." <<std::endl;
      return(false);
    }
    PyObject *pyv, *val;
    for (size_t i=0; i<numDerivVars; ++i) {
      pyv = PyList_GetItem(pym, i);
      if (!PyList_Check(pyv) || PyList_Size(pyv) != numDerivVars) {
	Cerr << "Python vector must have length " << numDerivVars << "." 
	     << std::endl;
	return(false);
      }
      for (int j=0; j<=i; ++j) {
	val = PyList_GetItem(pyv, j);
	if (PyFloat_Check(val))
	  rm(i,j) = PyFloat_AsDouble(val);
	else if (PyInt_Check(val))
	  rm(i,j) = (double) PyInt_AsLong(val);
	else {
	  Cerr << "Unsupported Python data type converting vector." 
	       << std::endl;
	  Py_DECREF(val);
	  return(false);
	}
      }
    }

  }
  return(true);
}


// assume numFns x numDerivVars x numDerivVars
// returns false if conversion failed
bool NRELPythonInterface::
python_convert(PyObject *pyma, RealSymMatrixArray &rma)
{
#ifdef DAKOTA_PYTHON_NUMPY
  if (userNumpyFlag) {
    // cannot recurse in this case as we now have a symmetric matrix 
    // (clearer this way anyway)
    if (!PyArray_Check(pyma) || PyArray_NDIM(pyma) != 3 || 
	PyArray_DIM(pyma,0) != numFns || PyArray_DIM(pyma,1) != numDerivVars ||
	PyArray_DIM(pyma,2) != numDerivVars ) {
      Cerr << "Python numpy array not 3D of size " << numFns << "x"
	   << numDerivVars << "x" << numDerivVars << "." << std::endl;
      return(false);
    }
    PyArrayObject *pao = (PyArrayObject *) pyma;
    for (size_t i=0; i<numFns; ++i)
      for (size_t j=0; j<numDerivVars; ++j)
	for (size_t k=0; k<=j; ++k)
	  rma[i](j,k) = *(double *)(pao->data + i*(pao->strides[0]) + 
				    j*(pao->strides[1]) +
				    k*(pao->strides[2]));
  }
  else
#endif
  {
    PyObject *val;
    if (!PyList_Check(pyma) || PyList_Size(pyma) != numFns) {
      Cerr << "Python matrix array must have " << numFns << " rows."
	   << std::endl;
      return(false);
    }
    for (size_t i=0; i<numFns; ++i) {
      val = PyList_GetItem(pyma, i);
      if (PyList_Check(val)) {
	if (!python_convert(val, rma[i]))
	  return(false);
      }
      else {
	Cerr << "Each row of Python matrix must be a list." << std::endl;
	Py_DECREF(val);
	return(false);
      }
    }
  }
  return(true);
}


} //namespace Dakota
