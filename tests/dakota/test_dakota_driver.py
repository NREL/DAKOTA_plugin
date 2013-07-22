
#import twister.interfaces.dakota._dakota as _dakota

from openmdao.main.api import Component, Assembly, set_as_top
from twister.assemblies.aep_csm_assembly import aep_csm_assembly
from twister.interfaces.dakota.dakota_driver import DAKOTADriver, DAKOTAInput



class AEP_CSM_DAKOTA_Scanner(Assembly):
    """Test assembly that creates a DAKOTA driver and runs a simple aep scan"""

    def __init__(self, nx):
        """ Creates a new Assembly containing an aep component and a DAKOTA scanner"""
        super(AEP_CSM_DAKOTA_Scanner, self).__init__()
        print "initializing AEP CSM DAKOTA scanner"
        self.infile = "dakota.aepcsm.in"

        # Create DAKOTA Optimizer instance
        driver = DAKOTADriver(self.infile)
        self.add('driver', driver)

        # Create Paraboloid component instances
        self.add('aepcomp', aep_csm_assembly()) 

        # Driver process definition
        self.driver.workflow.add('aepcomp')

        # DAKOTA input file
#        write_input_X(nx, "dakota.in.generic.template", self.infile)
        inp_dat = DAKOTAInput(
            method = ["multidim_parameter_study", "partitions = %d %d" % (nx, nx)],
            variables = [ "continuous_design = 2", "lower_bounds    4     1.3", "upper_bounds     15      3.5", "descriptors      'windSpeed50m'     'weibullK'"]
            )
        inp_dat.write_input(self.infile)

        # WTPerf/DaKOTA Objective
        self.driver.add_objective('aepcomp.aep')

        # WTPERF test Design Variables
        self.driver.add_parameter('aepcomp.windSpeed50m', low=4, high=15)
        self.driver.add_parameter('aepcomp.weibullK', low=1.4, high=3.5)



def test_rundak_driver():
    """ run dakota on an aep scan, test case for full dakota/openmdao/twister pipeline"""
    print "rundak_driver"
    dak = AEP_CSM_DAKOTA_Scanner(10)
    set_as_top(dak)
    dak.run()


if __name__=="__main__":
    ### make an openMDAO-based DAKOTA driver and test that by doing
    # scan of aep_csm component
    test_rundak_driver()
