FILE_NAME="build_gpu"
if [ ! -d $FILE_NAME ];then
  mkdir $FILE_NAME
fi
cd $FILE_NAME
cmake --log-level=ERROR -Wno-dev .. \
-DTPL_ENABLE_PARMETISLIB=TRUE \
-DTPL_PARMETIS_INCLUDE_DIRS="$PATH_TO_OPENBLAS_INC;$PATH_TO_PARMETIS_I32_INC;$PATH_TO_METIS_I32_INC" \
-DTPL_PARMETIS_LIBRARIES="$PATH_TO_PARMETIS_I32_LIB/libparmetis.so;$PATH_TO_METIS_I32_LIB/libmetis.so" \
-DTPL_BLAS_LIBRARIES="$PATH_TO_OPENBLAS_LIB/libopenblas.a" \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_Fortran_COMPILER=mpif77 \
-DTPL_ENABLE_CUDALIB=TRUE \
-DCUDA_LIBRARIES="$PATH_TO_CUDA_LIB/libcublas.so;$PATH_TO_CUDA_LIB/libcudart.so"

cd - > /dev/null
mkdir -p ppopp_output
cd $FILE_NAME/EXAMPLE
ln -s ../../ppopp_output
cd - > /dev/null

cd build_gpu/
make -j > /dev/null
cd - > /dev/null
