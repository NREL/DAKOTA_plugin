 Copyright 2013 National Renewable Energy Laboratory (NREL)
 
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 
 ++==++==++==++==++==++==++==++==++==++==
Test files for NREL DAKOTADriver openMDAO Driver
===============================================

Feb, 2013, Peter Graf

Brief summary of test materials for DAKOTADriver object:

Basic test, openMDAO NOT required:
---------------------------------
test_dakface.py: fires up dakota, passes a simple object, verifies that the object comes back through DAKOTA.

test_dakface.in: Dakota input file for this test.

Test of the openMDAO Driver:
---------------------------------
test_dakota_driver.py: runs a parameter scan of cost of energy, using one of our models.  You will maybe want to use 
a different model for your more generic openMDAO-related tests.

dakota.aepcsm.in: Dakota input file for the test.  This file is actuall written _by_ the test, so not needed, but may be a 
useful reference.


