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
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
#include <windows.h>
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

#include "mpi.h"

#include "DirectApplicInterface.H"


#include <Python.h>
//#include <arrayobject.h>
#include "dakface.h"


int main(int argc, char* argv[])
{
  int N = 10;
  double *some_dat = (double *)malloc(N * sizeof(double));
  int i;
  MPI_Comm comm = MPI_COMM_WORLD;

  for (i=0;i<N; i++)
    some_dat[i] = i;
  
  printf ("start here!\n");
  if (!Py_IsInitialized())
    {
      printf ("initializing python here in tdakota.main\n");
      Py_Initialize();
    }

  //  all_but_actual_main_core(argc, argv, &comm, some_dat);
  all_but_actual_main_core(argc, argv, &comm, NULL);
  free(some_dat);

    if (Py_IsInitialized()) {
      Py_Finalize();
      printf ("finalized python\n");
    }

  return 0;
}
