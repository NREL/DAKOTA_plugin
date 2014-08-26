# Copyright 2013 National Renewable Energy Laboratory (NREL)
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# 
# ++==++==++==++==++==++==++==++==++==++==
"""
Generic DAKOTA driver.
This uses the standard version of DAKOTA as a library (libdakota_src.so).
We have then implemented a custom interface (derived from DirectApplicInterface)
in C++.  This interface allows passing an mpi communicator and a void *.
Then we have wrapped that in boost-python.  The python call
accepts a boost.mpi MPI communicator and a generic python object that the C++
treats as a C "void *", then passes back to python.  There is no other
information passed to DAKOTA, so DAKOTA otherwise acts like the command line
version, in particular, all other inputs go through the input file.
"""

from __future__ import with_statement

import logging
import os
import sys

# Needed to avoid a problem with 'locale::facet::_S_create_c_locale'
if sys.platform in ('cygwin', 'win32'):
    os.environ['LC_ALL'] = 'C'

import pyDAKOTA

# Hard-coded assumption regarding availability of MPI.
if sys.platform in ('cygwin', 'win32'):
    _HAVE_MPI = False
else:
    _HAVE_MPI = True


class DakotaInput(object):
    """ simple mechanism where we store the actual strings that will go in each 
    section of the input file
        # provide your own input with key word arguments,
        # e.g.: DakotaInput(method = ["multidim_parameter_study", "partitions = %d %d" % (nx, nx)])
        """
    def __init__(self, **kwargs):
        self.strategy = [
            "single_method",
            "    tabular_graphics_data",
        ]
        self.method = [
            "multidim_parameter_study",
            "    partitions = 4 4",
        ]
        self.model = [
            "single",
        ]
        self.variables = [
            "continuous_design = 2",
            "    lower_bounds    3    5",
            "    upper_bounds    4    6",
            "    descriptors   'x1' 'x2'",
        ]	        
        self.interface = [
            "direct",
            "    analysis_drivers = 'NRELpython'",
            "    analysis_components = 'dakota:dakota_callback'",
        ]
        self.responses = [
            "num_objective_functions = 1",
            "no_gradients",
            "no_hessians",
        ]
        
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def write_input(self, infile):
        """ Write input file sections in standard order. """
        with open(infile, 'w') as out:
            for section in ('strategy', 'method', 'model', 'variables',
                            'interface', 'responses'):
                out.write('%s\n' % section)
                for line in getattr(self, section):
                    out.write("\t%s\n" % line)


class _ExcInfo(object):
    """ Used to hold exception return information. """

    def __init__(self):
        self.type = None
        self.value = None
        self.traceback = None


def run_dakota(infile, data=None, mpi_comm=None, stdout=None, stderr=None):
    """
    Run DAKOTA with `infile`.

    If `data` is not None, that is assumed to be a driver object supporting
    ``dakota_callback(**kwargs)``.

    If `mpi_comm` is not None, that is used as an MPI communicator.
    Otherwise, the ``world`` communicator from :class:`boost.mpi` is used
    if MPI is supported.

    `stdout` and `stderr` can be used to direct their respective DAKOTA
    stream to a filename.
    """

    # Checking for a Python exception via sys.exc_info() doesn't work, for
    # some reason it always returns (None, None, None).  So instead we pass
    # an object down and if an exception is thrown, the C++ level will fill
    # it with the exception information so we can re-raise it.
    err = 0
    exc = _ExcInfo()

    if data is None:
        if mpi_comm is None:
            if _HAVE_MPI:
                from boost.mpi import world
                err = pyDAKOTA.run_dakota_mpi(infile, world, stdout, stderr, exc)
            else:
                err = pyDAKOTA.run_dakota(infile, stdout, stderr, exc)
        else:
            err = pyDAKOTA.run_dakota_mpi(infile, mpi_comm, stdout, stderr, exc)
    else:
        if mpi_comm is None:
            if _HAVE_MPI:
                from boost.mpi import world
                err = pyDAKOTA.run_dakota_mpi_data(infile, world, data, stdout, stderr, exc)
            else:
                err = pyDAKOTA.run_dakota_data(infile, data, stdout, stderr, exc)
        else:
            err = pyDAKOTA.run_dakota_mpi_data(infile, mpi_comm, data, stdout, stderr, exc)

    # Check for errors. We'll only get here if DAKOTA has been updated to allow
    # aborts to return (throw an exception) rather than shut down the process.
    if err:
        if exc.type is None:
            raise RuntimeError('DAKOTA analysis failed')
        else:
            raise exc.type, exc.value, exc.traceback


def dakota_callback(**kwargs):    
    """
    Generic callback from DAKOTA, forwards parameters to driver provided as
    the ``data`` argument to :meth:`run_dakota`.

    The driver should return a responses dictionary from the parameters.
    `kwargs` contains:

    ========== ==============================================
    Key        Definition
    ========== ==============================================
    functions  number of functions (responses, constraints)
    ---------- ----------------------------------------------
    variables  total number of variables
    ---------- ----------------------------------------------
    cv         list/array of continuous variable values
    ---------- ----------------------------------------------
    div        list/array of discrete integer variable values
    ---------- ----------------------------------------------
    drv        list/array of discrete real variable values
    ---------- ----------------------------------------------
    av         single list/array of all variable values
    ---------- ----------------------------------------------
    cv_labels  continuous variable labels
    ---------- ----------------------------------------------
    div_labels discrete integer variable labels
    ---------- ----------------------------------------------
    drv_labels discrete real variable labels
    ---------- ----------------------------------------------
    av_labels  all variable labels
    ---------- ----------------------------------------------
    asv        active set vector (bit1=f, bit2=df, bit3=d^2f)
    ---------- ----------------------------------------------
    dvv        derivative variables vector
    ---------- ----------------------------------------------
    currEvalId current evaluation ID number
    ---------- ----------------------------------------------
    user_data  this object
    ========== ==============================================

    """
    if 'user_data' in kwargs:        
        driver = kwargs['user_data']
        return driver.dakota_callback(**kwargs)
    else:
        msg = '%s: no user_data passed to dakota_callback' % os.getpid()
        logging.error(msg)
        raise RuntimeError(msg)

