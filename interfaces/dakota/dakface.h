#ifndef _DAKFACE_H_
#define _DAKFACE_H_

#include "system_defs.h"
#include "global_defs.h"

#include "ParallelLibrary.H"
#include "ProblemDescDB.H"
#include "DakotaStrategy.H"
#include "DakotaModel.H"
#include "DakotaInterface.H"
//#include "PluginSerialDirectApplicInterface.H"
//#include "PluginParallelDirectApplicInterface.H"


// eventually use only _WIN32 here
/*
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
#include <windows.h>
#endif
*/
#include "ParallelLibrary.H"
#include "CommandLineHandler.H"
#include "ProblemDescDB.H"
#include "DakotaStrategy.H"
#ifdef DAKOTA_TRACKING
#include "TrackerHTTP.H"
#endif

//#define MPI_DEBUG
#if defined(MPI_DEBUG) && defined(MPICH2)
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include "DirectApplicInterface.H"

#ifndef WINDOWS
#include "mpi.h"
#else
#define MPI_Comm int
#define MPI_COMM_WORLD 0
#endif

#include <Python.h>
//#include <arrayobject.h>


int all_but_actual_main_mpi_data(int argc, char* argv[], MPI_Comm comm, void *data);
int all_but_actual_main(int argc, char* argv[]);
int all_but_actual_main_core(int argc, char* argv[], MPI_Comm *pcomm, void *data);

//using namespace Dakota;
namespace Dakota
//namespace NREL 
{
  class NRELApplicInterface : public DirectApplicInterface
  {
  public:
    NRELApplicInterface(const ProblemDescDB& problem_db, void *pData);
    ~NRELApplicInterface();
    void *pUserData;
  protected:
    // Virtual function redefinitions
    int derived_map_ac(const Dakota::String& ac_name);
  };

  class NRELPythonApplicInterface : public NRELApplicInterface
  {
  public:
    NRELPythonApplicInterface(const ProblemDescDB& problem_db, void *pData);
    ~NRELPythonApplicInterface();
  protected:
    // Virtual function redefinitions
    int derived_map_ac(const Dakota::String& ac_name);

    int userNumpyFlag;
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

  };
} // namespace
#endif
