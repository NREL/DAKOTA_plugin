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



