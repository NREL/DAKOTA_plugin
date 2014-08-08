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
Trivial pyDAKOTA test which runs a study on a 'model' which is the
Rosenbrock function.
"""

from numpy import array

from dakota import DakotaInput, run_dakota


class TestDriver(object):

    def __init__(self):
        self.input = DakotaInput()
        self.input.method = [
            "optpp_newton",
            "  max_iterations = 50",
            "  convergence_tolerance = 1e-4",
        ]
        self.input.variables = [
            "continuous_design = 2",
            "  cdv_initial_point  -1.2  1.0",
            "  cdv_lower_bounds   -2.0 -2.0",
            "  cdv_upper_bounds    2.0  2.0",
            "  cdv_descriptor      'x1' 'x2'",
        ]
        self.input.responses = [
            "num_objective_functions = 1",
            "analytic_gradients",
            "analytic_hessians",
        ]

    def run_dakota(self):
        """
        Call DAKOTA, providing self as `data`.
        DAKOTA will then call our :meth:`dakota_callback` during the run.
        """
        infile = 'test_driver.in'
        self.input.write_input(infile)
        run_dakota(infile, data=self)

    def dakota_callback(self, **kwargs):
        """
        Return responses from parameters.  `kwargs` contains:

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
        print 'dakota_callback:'
        cv = kwargs['cv']
        asv = kwargs['asv']
        print '    cv', cv
        print '    asv', asv

        # Rosenbrock function.
        x = cv
        f0 = x[1]-x[0]*x[0]
        f1 = 1-x[0]

        retval = dict()
        try:
            if asv[0] & 1:
                f = array([100*f0*f0+f1*f1])
                retval['fns'] = f

            if asv[0] & 2:
                g = array([[-400*f0*x[0] - 2*f1, 200*f0]])
                retval['fnGrads'] = g

            if asv[0] & 4:
                fx = x[1]-3*x[0]*x[0]
                h = array([ [ [-400*fx + 2, -400*x[0]],
                              [-400*x[0],    200     ] ] ] )
                retval['fnHessians'] = h

        except Exception, exc:
            print '    caught', exc
            raise

        print '    returning', retval
        return retval


if __name__ == '__main__':
    driver = TestDriver()
    driver.run_dakota()

