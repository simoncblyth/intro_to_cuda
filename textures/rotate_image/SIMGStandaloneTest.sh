#!/bin/bash -l 

name=SIMGStandaloneTest 

# attempt to suppress "was set but never used" warnings
# from complilation of stb_image.h using the below causing error 
# -Xcudafe "–-diag_suppress=set_but_not_used" 

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

nvcc $name.cu -lstdc++ -std=c++11  -I. -I${CUDA_PREFIX}/include -L${CUDA_PREFIX}/lib -lcudart -o /tmp/$name 
[ $? -ne 0 ] && echo compile FAIL && exit 1

/tmp/$name $*
[ $? -ne 0 ] && echo run FAIL && exit 2

exit 0 

