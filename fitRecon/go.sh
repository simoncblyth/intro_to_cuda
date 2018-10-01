#!/bin/bash -l

itc-
itc-cmake
itc-make


if [ "$(uname)" == "Darwin" ]; then
   otool-
   otool-rpath $(itc-bdir)/fitRecon
   otool-rpath $(itc-prefix)/lib/fitRecon
fi 


$(itc-prefix)/lib/fitRecon



