#CC = /opt/local/bin/g++-mp-4.5 
#CC = mpicc
####CC = /Users/pgraf/opt/mpich2-1.2.1p1.gfortran-mp-4.5/bin/mpic++
#CC = gcc
#CXX = /opt/local/bin/g++-mp-4.5 
####CXX = /Users/pgraf/opt/mpich2-1.2.1p1.gfortran-mp-4.5/bin/mpic++
#CXX = g++
#CXX = g++-4.2
#CXX = gcc-mp-4.3
CXX = mpic++

LD = ld

INCLUDE_DIR = -I/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy/core/include/ -I/opt/local/Library/Frameworks/Python.framework//Versions/2.6/include/python2.6 -I/Users/pgraf/root/include -I/Users/pgraf/root/dakota/include/

DAK_INCDIR = -I/Users/pgraf/opt/Dakota/Dakota_NREL//packages/nidr -I/Users/pgraf/opt/Dakota/Dakota_NREL/ -I/Users/pgraf/opt/Dakota/Dakota_NREL/include

LIB_DIR = -L/Users/pgraf/root/lib 
LIB_PY = /opt/local/Library/Frameworks/Python.framework/Versions/2.6/Python
LIBS =  -lboost_python -lboost_mpi -lboost_system

# -lpython2.6
# -L/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/config

LIBMODEOBJ = /Users/pgraf/opt/Dakota/Dakota_NREL/src/library_mode.o
#LIBMODEOBJ = 

#COMPILER_LIBS = -L/opt/local/lib/gcc45/gcc/i386-apple-darwin9/4.5.0 -L/opt/local/lib/gcc45/gcc/i386-apple-darwin9/4.5.0/../../.. /opt/local/lib/gcc45/libgfortran.dylib -lm -lgcc_s.10.5 -lgcc_ext.10.5 -lSystem -ldl

COMPILER_LIBS = -L/usr/lib/gcc/i686-apple-darwin9/4.0.1/ -lm -lgcc  -lSystem -ldl   -L/opt/local/lib/gcc45/gcc/i386-apple-darwin9/4.5.0/../../.. /opt/local/lib/gcc45/libgfortran.dylib  -lgcc_ext.10.5


DAKOTA_LIBS = -L/Users/pgraf/root/dakota/lib/ -ldakota -llhs -lnewmat -lnidr -lopt -lpecos -lconmin -lsparsegrid -lteuchos 
DAKOTA = $(DAKOTA_LIBS) -L/usr/X11/lib -lSM -lICE -llapack -lblas $(COMPILER_LIBS)

#-L/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/config -lpython2.6 -ldl

OBJS_BASE = python_interface.o dakface.o
OBJS_TDAKOTA = $(OBJS_BASE) tdakota_main.o
OBJS_DAKOTA = dakota_python_binding.o $(OBJS_BASE) 
OBJS_OLDDAKOTA = dakota_python_binding.o  $(LIBMODEOBJ) 
OBJS_TESTBINDING = test_python_binding.o

FFLAGS = -g
CFLAGS = -g -DHAVE_MPI -DDAKOTA_PYTHON_NUMPY
TESTCFLAGS = -g 
LIBLDFLAGS = -g 
LIBLDFLAGSDYLD = -g -dynamiclib
LDFLAGS = -g 

#NOTE: twisted_dakota need TDAKOTA so a main() gets built, and so python is initialized

#all: twisted_dakota
#all: twisted_dakota 
#all: twisted_dakota  dakota_py test_binding_py
all: dakota_py test_binding_py
#old_dakota_py

twisted_dakota: $(OBJS_TDAKOTA)
	$(CXX) $(LIBLDFLAGS) $(DAKOTA) $(OBJS_TDAKOTA) $(LIB_DIR)  $(LIB_PY) $(LIBS)  -o tdakota

dakota_py: $(OBJS_DAKOTA)
	$(CXX) $(LIBLDFLAGSDYLD) $(DAKOTA) $(OBJS_DAKOTA) $(LIB_DIR) $(LIB_PY) $(LIBS)  -o _dakota.so

old_dakota_py: $(OBJS_OLDDAKOTA)
	$(CXX) $(LIBLDFLAGSDYLD) $(DAKOTA) $(OBJS_OLDDAKOTA) $(LIB_DIR) $(LIB_PY) $(LIBS)  -o _olddakota.so

test_binding_py: $(OBJS_TESTBINDING)
	$(CXX) $(LIBLDFLAGSDYLD) $(LIB_DIR) $(LIB_PY)  $(OBJS_TESTBINDING) $(LIBS)  -o _testbinding.so


.f.o:
	$(F77) $(INCLUDE_DIR) $(FFLAGS) -c $< -o $@
.c.o:
	$(CC)  $(INCLUDE_DIR) $(CFLAGS) -c $< -o $@
.cpp.o:
	$(CXX)  $(INCLUDE_DIR) $(CFLAGS) -c $< -o $@
.cc.o:
	$(CXX)  $(INCLUDE_DIR) $(CFLAGS)   -c $< -o $@

clean:
	rm -f *.o

