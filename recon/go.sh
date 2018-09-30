#!/bin/bash -l

itc-

sdir=$(pwd)
base=$(itc-prefix)

bdir=$base/build/$(basename $sdir)
idir=$base

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_PREFIX_PATH=$(itc-prefix)/externals \
     -DCMAKE_INSTALL_PREFIX=$idir 


make VERBOSE=1
make install VERBOSE=1  


if [ "$(uname)" == "Darwin" ]; then
   otool-
   otool-rpath $bdir/tests/ReconTest
   otool-rpath $idir/lib/ReconTest
fi 

