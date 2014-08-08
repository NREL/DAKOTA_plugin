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

/*
This is basically identical to PythonInterface, except:

1. It passes a user data pointer to the invoked Python function.
2. It can let Python exceptions propagate back to a Python caller of
   DAKOTA if other parts of DAKOTA have been updated to support this.
3. The constructor handles Python interpreter initialization better, in that if
   the intepreter is already running, we don't try to initialize it again.
   The interpreter would already be running if DAKOTA was invoked by a
   Python function.

This last point is why we can't simply inherit from PythonInterface.
Sadly, we have to duplicate it.
*/

#ifndef NREL_PYTHON_INTERFACE_H
#define NREL_PYTHON_INTERFACE_H

#include "DirectApplicInterface.hpp"

// The following to forward declare, but avoid clash with include
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

namespace Dakota {

/** Specialization of DirectApplicInterface to link to Python analysis 
    drivers.  Includes convenience functions to map data to/from Python */

class NRELPythonInterface: public DirectApplicInterface
{

public:

  /// constructor
  NRELPythonInterface(const ProblemDescDB& problem_db, void *pData=0);
  /// destructor
  ~NRELPythonInterface();

protected:

  /// execute an analysis code portion of a direct evaluation invocation
  virtual int derived_map_ac(const String& ac_name);

  /// direct interface to Python via API, BMA 07/02/07
  int python_run(const String& ac_name);
  /// whether the user requested numpy data structures
  bool userNumpyFlag;
  /// convert arrays of integer types to Python list or numpy array
  template<class ArrayT, class Size>
  bool python_convert_int(const ArrayT& src, Size size, PyObject** dst);
  /// convert RealVector to Python list or numpy array
  bool python_convert(const RealVector& src, PyObject** dst);
  /// convert RealVector + IntVector + RealVector to Python mixed list 
  /// or numpy double array
  bool python_convert(const RealVector& c_src, const IntVector& di_src,
                      const RealVector& dr_src, PyObject** dst);
  /// convert labels
  bool python_convert(const StringMultiArray& src, PyObject** dst);
  /// convert all labels to single list
  bool python_convert(const StringMultiArray& c_src,
                      const StringMultiArray& di_src,
                      const StringMultiArray& dr_src, PyObject** dst);
  /// convert python [list of int or float] or [numpy array of double] to 
  /// RealVector (for fns)
  bool python_convert(PyObject *pyv, RealVector& rv, const int& dim);
  /// convert python [list of int or float] or [numpy array of double] to 
  /// double[], for use as helper in converting gradients
  bool python_convert(PyObject *pyv, double *rv, const int& dim);
  /// convert python [list of lists of int or float] or [numpy array of dbl]
  /// to RealMatrix (for gradients)
  bool python_convert(PyObject *pym, RealMatrix &rm);
  /// convert python [list of lists of int or float] or [numpy array of dbl]
  /// to RealMatrix (used as helper in Hessian conversion)
  bool python_convert(PyObject *pym, RealSymMatrix &rm);
  /// convert python [list of lists of lists of int or float] or 
  /// [numpy array of double] to RealSymMatrixArray (for Hessians)
  bool python_convert(PyObject *pyma, RealSymMatrixArray &rma);

private:

  /// generic data pointer
  void *_pUserData;

  /// did we create the interpreter?
  bool _myPython;
};

} // namespace Dakota

#endif  // NREL_PYTHON_INTERFACE_H
