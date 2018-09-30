#!/bin/bash -l

itc-
itc-bwipe
itc-cmake
itc-make

if [ "$(uname)" == "Darwin" ]; then
   otool-
   otool-rpath $(itc-bdir)/tests/ReconTest
   otool-rpath $(itc-prefix)/lib/ReconTest
fi 

