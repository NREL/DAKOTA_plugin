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


