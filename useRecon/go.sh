#!/bin/bash -l

itc-
itc-cmake
itc-make


if [ "$(uname)" == "Darwin" ]; then
   otool-
   otool-rpath $(itc-bdir)/useRecon
   otool-rpath $(itc-prefix)/lib/useRecon
fi 


$(itc-prefix)/lib/useRecon



