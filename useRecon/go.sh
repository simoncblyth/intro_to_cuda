#!/bin/bash -l

itc-

sdir=$(pwd)
bdir=$(itc-bdir $sdir)

printf " sdir %50s : bdir %50s  \n " $sdir $bdir


itc-cmake
itc-make

if [ "$(uname)" == "Darwin" ]; then

   otool-
   otool-rpath $bdir/useRecon
   otool-rpath $(itc-prefix)/lib/useRecon

fi 
