"""
# generic DAKOTA driver
# This uses the standard version of DAKOTA as a library (libdakota.a).
# We have then implemented a custom interface (derived from DirectApplicInterface)
# in C++.  This interface allows passing and mpi communicator and a void *.
# Then we have wrapped that in boost-python.  The python call
# accepts a boost.mpi MPI communicator and a generic python object that the C++ treats as a
# C "void *", then passes back to python.  There is no other information passed
# to DAKOTA, so DAKOTA otherwise acts like the command line version, in particular,
# all other inputs go through the input file.
"""
import twister.interfaces.dakota._dakota as _dakota
        
from openmdao.main.driver import Driver
from openmdao.main.hasparameters import HasParameters
from openmdao.main.hasobjective import HasObjective
from openmdao.util.decorators import add_delegate

class DAKOTAInput(object):
    """ simple mechanism where we store the actual strings that will go in each 
    section of the input file
        # provide your own input with key word arguments,
        # e.g.: DAKOTAInput(method = ["multidim_parameter_study", "partitions = %d %d" % (nx, nx)])
        """
    def __init__(self, **kwargs):
        self.strategy = ["single_method", "tabular_graphics_data"]
        self.method = ["multidim_parameter_study", "partitions = 4 4"]
        self.model = ["single"]
        self.variables = [ "continuous_design = 2", "lower_bounds    3     5", "upper_bounds     4      6", "descriptors      'x1'     'x2'"]	        
        self.interface = ["direct", "analysis_drivers = 'NRELpython'",
                     "analysis_components = 'twister.interfaces.dakota.dakota_driver:dakota_callback'"]
        self.responses = ["num_objective_functions = 1" , "no_gradients", "no_hessians"]
        
        # provide your own input with key word arguments,
        # e.g.: method = ["multidim_parameter_study", "partitions = %d %d" % (nx, nx)]
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def write_input(self, infile):
        """ write input file using self.__dict__"""
        # PG gets fancy!
        f = file(infile, "w")
        for section in self.__dict__:
            f.write("%s\n" % section)
            for ln in self.__dict__[section]:
                f.write("\t%s\n" % ln)
        f.close()


@add_delegate(HasParameters,  HasObjective)
class DAKOTADriver(Driver):
    """ Driver wrapper of C++ DAKOTA.                    
    """
    __have_mpi__ = False

    def __init__(self, input_file, *args, **kwargs):
        super(DAKOTADriver, self).__init__(*args, **kwargs)
        
        self.iter_count = 0
        self.infile = input_file
        import logging
        self._logger._set_level(logging.DEBUG)
        
    def execute(self):
        """ call dakota, providing self as data. Callback will then call run_iteration for the driver """
        import twister.interfaces.dakota._dakota as _dakota
        # calls dakota with arguments that equal "dakota dakota.in"
        # import OUR boost-python binding for dakota, defined in dakota_python_binding.c,
        # which calls to OUR function all_but_actual_main() in Dakota/src/library_mode.C
        print "running DAKOTA from input file: ", self.infile
        if (self.__have_mpi__):
            from boost.mpi import world
            _dakota.run_dakota_mpi_data(self.infile, world, self)
        else:
            _dakota.run_dakota_data(self.infile, self)
    
    def run_iteration(self):
        """ The DAKOTA driver iteration.
        This is called from the callback from DAKOTA
        """

        self._logger.debug('iteration_count = %d' % self.iter_count)
        
        super(DAKOTADriver, self).run_iteration()
        # will execute the workflow
        
    def set_values(self,x):
        self.set_parameters(x)

    def get_responses(self):
        return self.eval_objective()
                    
# generic callback from dakota
# will be passed a Driver.  mission of this function
# is to set the passed in parameters in the associated workflow and then
# execute the workflow.
# goal is that this function be generic.

def dakota_callback(**kwargs):    
    from numpy import array
    print "entering dakota callback, really"

    x = kwargs['cv']
    ASV = kwargs['asv']
    # x = vector of input arguments
    # ASV = mask of what we want (bit 1 = f, bit 2 = df, bit3 = d^2f)

    print "x = ", x
    retval = dict([])
    retval['fns'] = array([0])

    if 'user_data' in kwargs:        
        driver = kwargs['user_data']
        print "I have  a driver:", driver
        # Driver had better be an instance of DAKOTADriver  !!!
        driver.set_values(x)
        print "calling run_iteration"
        driver.run_iteration()
        val = driver.get_responses()
        retval['fns'] = array([val])
    else:
        print "no driver passed to dakota callback"
        raise ValueError

    print "returning ", retval
    return retval


