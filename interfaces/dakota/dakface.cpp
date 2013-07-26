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
// 1) and MPI comm to work with
// 2) a void * to pass back to the evaluation function

// We will use DAKOTA in library mode as described at
// http://dakota.sandia.gov/docs/dakota/5.2/html-dev/DakLibrary.html

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
/*   CAUSES PROBLEM IN MINGW B/C "interface" becomes some sort of struct or keyword
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
#include <windows.h>
#endif
*/
/// BUT THen turns out we need it for gcc with non-cygwin python
#ifdef WINDOWS
#include <windows.h>
#ifdef interface 
#undef interface 
#endif
#endif

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

#ifndef WINDOWS
#include "mpi.h"
#endif

#include "DirectApplicInterface.H"


#include <Python.h>
//#include <arrayobject.h>
#include "dakface.h"


#ifdef HAVE_AMPL
/** Floating-point initialization from AMPL: switch to 53-bit rounding
    if appropriate, to eliminate some cross-platform differences. */
  extern "C" void fpinit_ASL();
#endif

//extern "C" int nidr_save_exedir(const char*, int);

//using namespace Dakota;

using namespace Dakota ;
//{
//namespace NREL {

  /****************************/
  ///// basic NREL Applic Interface.  
  /// main purpose is to accept and pass through a generic void *
  /// will be used by python, but could be used any other way too
  ////comm is already accepted in main(). coordinate that later.
  NRELApplicInterface::NRELApplicInterface(const ProblemDescDB& problem_db, void *pData) : DirectApplicInterface(problem_db)
  {
    pUserData = pData;
  }
  
  NRELApplicInterface::~NRELApplicInterface()
  {
  }

  int NRELApplicInterface::derived_map_ac(const Dakota::String& ac_name)
  {
    printf ("This is my chance to evaluate something, yeah!?\n");
    double *dat = (double *)pUserData;
    int i;
    fnVals[0] = 0;
    for (i=0;i<10;i++)
      fnVals[0] += dat[i] * xC[0] * xC[0];
    return 0;
    //    return (DirectApplicInterface::derived_map_ac(ac_name));
  }
//}  // namespace NREL

/**************************/


int all_but_actual_main_mpi_data(int argc, char* argv[], MPI_Comm comm, void *data)
{
  return (all_but_actual_main_core(argc, argv, &comm, data));
}

int all_but_actual_main(int argc, char* argv[])
{
  return (all_but_actual_main_core(argc, argv, NULL, NULL));
}


int all_but_actual_main_core(int argc, char* argv[], MPI_Comm *pcomm, void *data)
{
  printf ("REALLY entered DAKOTA main core\n");
  //  nidr_save_exedir(argv[0], 3);	// 3 ==> add both the directory containing this binary
				// and . to the end of $PATH if not already on $PATH.
  //    printf ("bail!\n"); return(0);

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
  char *pname;
  int dfd;
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

  /// manage "data" here: make an instance or OUR application interface subclass that can carry a void * around, and
  // calls the function we want to call
  //// TODOTODOTODO
  // retrieve the list of Models from the Strategy

  int interface;

  printf ("looking for interfaces to replace\n");
  ModelList& models = problem_db.model_list();
  // iterate over the Model list
  for (ModelLIter ml_iter = models.begin(); ml_iter != models.end(); ml_iter++) {
    Model mod = *ml_iter;
    bool force = mod.force_rebuild();
    Interface& interface = mod.interface();
    //    Interface& iface = ml_iter->interface();
    printf ("interface: %s %s\n", interface.interface_type().data(), interface.analysis_drivers()[0].data());
    if (interface.interface_type() == "direct")
      {
	if (contains(interface.analysis_drivers(),"NREL"))
	  {
	    printf ("replacing interface with NRELApplicInterface\n");
	    // set the correct list nodes within the DB prior to new instantiations
	    problem_db.set_db_model_nodes(ml_iter->model_id());
	    // plug in the new direct interface instance
	    //	    interface.assign_rep(new NREL::NRELApplicInterface(problem_db, data), false);
	    interface.assign_rep(new NRELApplicInterface(problem_db, data), false);
	  }
	else if  (contains(interface.analysis_drivers(),"NRELpython"))
	  {
	    printf ("replacing interface with NRELPythonApplicInterface\n");
	    // set the correct list nodes within the DB prior to new instantiations
	    problem_db.set_db_model_nodes(ml_iter->model_id());
	    // plug in the new direct interface instance
	    //	    interface.assign_rep(new NREL::NRELPythonApplicInterface(problem_db, data), false);
	    interface.assign_rep(new NRELPythonApplicInterface(problem_db, data), false);
	  }
      }
  }
  

#ifdef DAKOTA_TRACKING
  // must wait for iterators to be instantiated; positive side effect is that 
  // we don't track dakota -version, -help, and errant usage
  TrackerHTTP usage_tracker(problem_db, parallel_lib->world_rank());
  usage_tracker.post_start();
#endif



  // Optionally run the strategy
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
    selected_strategy.run_strategy();
    printf ("strategy done!\n");
    std::signal(SIGCHLD, sigchld_save);
    printf ("strategy really done!\n");
#endif
  }

#ifdef DAKOTA_TRACKING
  usage_tracker.post_finish();
#endif

  printf ("trying to escape this program!!\n");
  return 0;
}

//}  // end namespace Dakota
