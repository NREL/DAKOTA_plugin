import twister.interfaces.dakota._dakota as _dakota

class AnObj(object):
    """ test object to show we can pass an object through our DAKOTA interface """
    def __init__(self):
        self.mynum = 10

def callback(**kwargs):
    """ callback for rosenbrock test case """
    if ('user_data' in kwargs):
        obj = kwargs['user_data']
        print "magic number is ", obj.mynum

    from numpy import array

    num_fns = kwargs['functions']
        
    x = kwargs['cv']
    ASV = kwargs['asv']

    f0 = x[1]-x[0]*x[0]
    f1 = 1-x[0]

    retval = dict([])

    if (ASV[0] & 1): # **** f:
        f = array([100*f0*f0+f1*f1])
        retval['fns'] = f

    if (ASV[0] & 2): # **** df/dx:
        g = array([[-400*f0*x[0] - 2*f1, 200*f0]])
        retval['fnGrads'] = g

    if (ASV[0] & 4): # **** d^2f/dx^2:
        fx = x[1]-3*x[0]*x[0]
        
        h = array([ [ [-400*fx + 2, -400*x[0]],
              [-400*x[0],    200     ] ] ]    )
        retval['fnHessians'] = h

    return(retval)

def test_rundak():
    """ run dakota to solve rosenbrock all driven from python """
    __have_mpi__ = False
    if (not __have_mpi__):
        _dakota.run_dakota_data("test_dakface.in", AnObj())
    else:
        from boost.mpi import world
        _dakota.run_dakota_mpi_data("test_dakface.in", world, AnObj())
    print "made it back from dakota call"
    
if __name__=="__main__":
    """ super simple test of connectivity. solves Rosenbrock, no openMDAO.
    requires test_dakface.in to be present """
    test_rundak()

