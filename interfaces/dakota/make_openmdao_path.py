

import os,sys,fileinput

from distutils.sysconfig import get_python_lib
prefix = get_python_lib()
#print prefix

files = os.listdir(prefix)
#print files

str = "export PYTHONPATH=$PYTHONPATH"
for ln in files:
    if (len(ln) > 8):
        path = os.path.join(prefix, ln)
        str += ":%s" % path
    
#    str += ":/Users/pgraf/work/wese/wese/"
f = file("setopenmdaopythonpath.sh", "w")
f.write(str)
f.close

print "run: 'source setopenmdaopythonpath.sh' to setup python paths for DAKOTA to import modules that import openMDAO modules"



