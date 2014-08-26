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
// Peter Graf, 9/21/12
// Implementing a C++ application interface for Dakota that will accept
// 1) an MPI comm to work with
// 2) a void * to pass back to the evaluation function

// We will use DAKOTA in library mode as described at
// http://dakota.sandia.gov/docs/dakota/5.2/html-dev/DakLibrary.html

// Must be before system includes according to Python docs.
//#include <Python.h>
// Replaces Python.h according to boost_python docs.
#include <boost/python/detail/wrap_python.hpp>

#include "dakota_system_defs.hpp"
#include "dakota_global_defs.hpp"

#include "DakotaModel.hpp"
#include "DakotaInterface.hpp"

// eventually use only _WIN32 here
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
#include <windows.h>
#ifdef interface 
#undef interface 
#endif
#endif

#include "ParallelLibrary.hpp"
#include "CommandLineHandler.hpp"
#include "ProblemDescDB.hpp"
#include "DakotaStrategy.hpp"
#ifdef DAKOTA_USAGE_TRACKING
#include "TrackerHTTP.hpp"
#endif

//#define MPI_DEBUG
#if defined(MPI_DEBUG) && defined(MPICH2)
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include "dakface.hpp"
#include "python_interface.hpp"

#include <boost/python/def.hpp>
namespace bp = boost::python;

#ifdef HAVE_AMPL
/** Floating-point initialization from AMPL: switch to 53-bit rounding
    if appropriate, to eliminate some cross-platform differences. */
  extern "C" void fpinit_ASL();
#endif

//extern "C" int nidr_save_exedir(const char*, int);

using namespace Dakota;

static int _main(int argc, char* argv[], MPI_Comm *pcomm, void *data, void *exc);

int all_but_actual_main(int argc, char* argv[], void *exc)
{
  return _main(argc, argv, NULL, NULL, exc);
}

int all_but_actual_main_mpi(int argc, char* argv[],
                            MPI_Comm comm, void *exc)
{
  return _main(argc, argv, &comm, NULL, exc);
}

int all_but_actual_main_mpi_data(int argc, char* argv[],
                                 MPI_Comm comm, void *data, void *exc)
{
  return _main(argc, argv, &comm, data, exc);
}

int all_but_actual_main_data(int argc, char* argv[],
                             void *data, void *exc)
{
  return _main(argc, argv, NULL, data, exc);
}


static int _main(int argc, char* argv[], MPI_Comm *pcomm, void *data, void *exc)
{
  // 3 ==> add both the directory containing this binary and . to the end
  // of $PATH if not already on $PATH.
//  nidr_save_exedir(argv[0], 3);

#ifdef HAVE_AMPL
  fpinit_ASL();	// Switch to 53-bit rounding if appropriate, to
		// eliminate some cross-platform differences.
#endif
/*
#ifdef __MINGW32__
  std::signal(WM_QUIT, abort_handler);
  std::signal(WM_CHAR, abort_handler);
#else
  std::signal(SIGKILL, abort_handler);
  std::signal(SIGTERM, abort_handler);
#endif
*/
  std::signal(SIGINT,  abort_handler);
#ifdef HAVE_ABORT_RETURNS
  abort_returns = true;  // abort_handler() will try to return to us rather
                         // than aborting the process.
#endif

#ifdef MPI_DEBUG
  // hold parallel job prior to MPI_Init() in order to attach debugger to
  // master process.  Then step past ParallelLibrary instantiation and attach
  // debugger to other processes.
#ifdef MPICH2
  // To use this approach, set $DAKOTA_DEBUGPIPE to a suitable name,
  // and create $DAKOTA_DEBUGPIPE by executing "mkfifo $DAKOTA_DEBUGPIPE".
  // After invoking "mpirun ... dakota ...", find the processes, invoke
  // a debugger on them, set breakpoints, and execute "echo >$DAKOTA_DEBUGPIPE"
  // to write something to $DAKOTA_DEBUGPIPE, thus releasing dakota from
  // a wait at the open invocation below.
  char *pname; int dfd;
  if ( ( pname = getenv("DAKOTA_DEBUGPIPE") ) &&
       ( dfd = open(pname,O_RDONLY) ) > 0 ) {
    char buf[80];
    read(dfd,buf,sizeof(buf));
    close(dfd);
  }
#else
  // This simple scheme has been observed to fail with MPICH2
  int test;
  std::cin >> test;
#endif // MPICH2
#endif // MPI_DEBUG

  // Instantiate/initialize the parallel library, command line handler, and
  // problem description database objects.  The ParallelLibrary constructor
  // calls MPI_Init() if a parallel launch is detected.  This must precede
  // CommandLineHandler initialization/parsing so that MPI may extract its
  // command line arguments first, prior to DAKOTA command line extractions.
  ParallelLibrary *parallel_lib;
  if (pcomm)
    parallel_lib = new ParallelLibrary(*pcomm);
  else
    parallel_lib = new ParallelLibrary(argc, argv);

  CommandLineHandler cmd_line_handler(argc, argv);
  // if just getting version or help, exit now
  if (!cmd_line_handler.instantiate_flag())
    return 0;

  ProblemDescDB problem_db(*parallel_lib);

  // Manage input file parsing, output redirection, and restart processing.
  // Since all processors need the database, manage_inputs() does not require
  // iterator partitions and it can precede init_iterator_communicators()
  // (a simple world bcast is sufficient).  Output/restart management does
  // utilize iterator partitions, so manage_outputs_restart() must follow
  // init_iterator_communicators() within the Strategy constructor
  // (output/restart options may only be specified at this time).
  parallel_lib->specify_outputs_restart(cmd_line_handler);
  // ProblemDescDB requires run mode information from the parallel
  // library, so must follow the specify outputs_restart
  problem_db.manage_inputs(cmd_line_handler);

  // Instantiate the Strategy object (which instantiates all Model and Iterator
  // objects) using the parsed information in problem_db.  All MPI communicator
  // partitions are created during strategy construction.
  Strategy selected_strategy(problem_db);

  // retrieve the list of Models from the Strategy
  ModelList& models = problem_db.model_list();
  // iterate over the Model list
  for (ModelLIter ml_iter = models.begin(); ml_iter != models.end(); ml_iter++) {
    Interface& interface = ml_iter->interface();
    if (interface.interface_type() == "direct") {
      if (contains(interface.analysis_drivers(), "NRELpython")) {
	// set the correct list nodes within the DB prior to new instantiations
	problem_db.set_db_model_nodes(ml_iter->model_id());
	// plug in the new direct interface instance
	interface.assign_rep(new NRELPythonInterface(problem_db, data), false);
      }
    }
  }
  
#ifdef DAKOTA_USAGE_TRACKING
  // must wait for iterators to be instantiated; positive side effect is that 
  // we don't track dakota -version, -help, and errant usage
  TrackerHTTP usage_tracker(problem_db, parallel_lib->world_rank());
  usage_tracker.post_start();
#endif

  // Optionally run the strategy
  int retval = 0;
  if (cmd_line_handler.retrieve("check")) {
    std::string check_msg("\nInput check completed successfully (input ");
    check_msg += "parsed and objects instantiated).\n";
    parallel_lib->output_helper(check_msg);  // rank 0 only
  }
  else {
    problem_db.lock(); // prevent run-time DB queries
#ifdef _WIN32
    selected_strategy.run_strategy();
#else
    // workaround: some callers of DAKOTA, e.g., mpirun might register a handler
    // for SIGCHLD, but never reset it, which interferes with our fork interface
    void (*sigchld_save)(int);
    sigchld_save = std::signal(SIGCHLD, SIG_DFL);
    try {
      selected_strategy.run_strategy();
    }
    catch (int fail_code) {
      retval = fail_code;
      if (PyErr_Occurred()) {
        PyObject *type = NULL, *value = NULL, *traceback = NULL;
        PyErr_Fetch(&type, &value, &traceback);
        bp::object *tmp = (bp::object *)exc;
        PyObject_SetAttrString(tmp->ptr(), "type", type);
        PyObject_SetAttrString(tmp->ptr(), "value", value);
        PyObject_SetAttrString(tmp->ptr(), "traceback", traceback);
      }
    }
    std::signal(SIGCHLD, sigchld_save);
#endif
  }

#ifdef DAKOTA_USAGE_TRACKING
  usage_tracker.post_finish();
#endif

  return retval;
}

