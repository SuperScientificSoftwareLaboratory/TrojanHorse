############################################################################
#
#  Program:         SuperLU_DIST
#
#  Module:          make.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   March 1, 2016	version 5.0.0
#
#  Modified:	    October 13, 2017    version 5.2.1
#		    February 20, 2021   version 7.0.0
#		    October 5, 2021     version 7.1.0
#
############################################################################
#
#  The name of the libraries to be created/linked to
#
SuperLUroot = /usr/local
#DSUPERLULIB = $(SuperLUroot)/SRC/libsuperlu_dist.a
#DSUPERLULIB = $(SuperLUroot)/lib/libsuperlu_dist.a
DSUPERLULIB = /usr/local/lib/libsuperlu_dist.a
INCLUDEDIR  = $(SuperLUroot)/include

XSDK_INDEX_SIZE = 
SLU_HAVE_LAPACK = FALSE
HAVE_PARMETIS   = TRUE
HAVE_COLAMD     = 
HAVE_COMBBLAS   = 
HAVE_CUDA       = TRUE
HAVE_HIP        = 

XSDK_ENABLE_Fortran = ON
ifeq ($(XSDK_ENABLE_Fortran),ON)
#  DFORTRANLIB = $(SuperLUroot)/lib/libsuperlu_dist_fortran.a
  DFORTRANLIB = /usr/local/lib/libsuperlu_dist_fortran.a
  LIBS = $(DFORTRANLIB) $(DSUPERLULIB) /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/openblas-0.3.28-6lswnbxtfkhopbtexopgepewwx3owmr6/lib/libopenblas.a -lm
  LIBS +=  -lgfortran -lm -lgcc_s -lgcc -lquadmath -lm -lc -lgcc_s -lgcc
else
  LIBS = $(DSUPERLULIB) /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/openblas-0.3.28-6lswnbxtfkhopbtexopgepewwx3owmr6/lib/libopenblas.a -lm
endif

LIBS	 += 
LIBS	 += /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/parmetis-4.0.3-skf7nuy2ssyzz3dguees2sqs77cwkalx/lib/libparmetis.so /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/metis-5.1.0-xt7fkius267tjbqtk2unyz4bf27er5ga/lib/libmetis.so
LIBS	 += 
LIBS 	 += 
LIBS 	 += -lgcc_s -lgcc -lc -lgcc_s -lgcc -ldl -lpthread -lrt -lmpi -lmpifort 
#LIBS     += CUDA::cudart CUDA::cublas CUDA::cusolver

CUDALIBS = /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/cuda-12.8.1-v7vnla6na7wldgwyguqimefqec2qrcos/lib64/libcudart.so /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/cuda-12.8.1-v7vnla6na7wldgwyguqimefqec2qrcos/lib64/libcublas.so /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/cuda-12.8.1-v7vnla6na7wldgwyguqimefqec2qrcos/lib64/libcusolver.so /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/cuda-12.8.1-v7vnla6na7wldgwyguqimefqec2qrcos/lib64/libcusparse.so
LIBS     += $(CUDALIBS)

#
#  The archiver and the flag(s) to use when building archive (library)

#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         = /usr/bin/ar
ARCHFLAGS    = cr
RANLIB       = /usr/bin/ranlib

CC           = /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/intel-oneapi-mpi-2021.17.0-qgsodl2pz3h436geq36wttdfb26xvwd3/mpi/2021.17/bin/mpicc
CFLAGS 	     = -O3 -DNDEBUG -I/root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/metis-5.1.0-xt7fkius267tjbqtk2unyz4bf27er5ga/include -I/root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/parmetis-4.0.3-skf7nuy2ssyzz3dguees2sqs77cwkalx/include -I/root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/openblas-0.3.28-6lswnbxtfkhopbtexopgepewwx3owmr6/include -DUSE_VENDOR_BLAS -fopenmp  -march=x86-64 -mtune=generic 
##-fPIC
#CFLAGS      += -D
# CFLAGS     += 
CXX          = /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/intel-oneapi-mpi-2021.17.0-qgsodl2pz3h436geq36wttdfb26xvwd3/mpi/2021.17/bin/mpicxx
CXXFLAGS     = -O3 -DNDEBUG -fopenmp  -march=x86-64 -mtune=generic
NVCC	     = /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/cuda-12.8.1-v7vnla6na7wldgwyguqimefqec2qrcos/bin/nvcc
NVCCFLAGS    = -DUSE_VENDOR_BLAS  -Wno-deprecated-gpu-targets -O3

NOOPTS       = -O0 -g
FORTRAN	     = /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/intel-oneapi-mpi-2021.17.0-qgsodl2pz3h436geq36wttdfb26xvwd3/mpi/2021.17/bin/mpif77
FFLAGS	     =  -march=x86-64 -mtune=generic 

LOADER       = /root/ppopp26trojanhorseae/spack/opt/spack/linux-x86_64/intel-oneapi-mpi-2021.17.0-qgsodl2pz3h436geq36wttdfb26xvwd3/mpi/2021.17/bin/mpicxx
LOADOPTS     =      
