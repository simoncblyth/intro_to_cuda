ibcm-source(){ echo $BASH_SOURCE ; }
ibcm-vi(){ vi $(ibcm-source) ; }
ibcm-env(){ echo -n ; } 
ibcm-usage(){ cat << EOU

Boost CMake Modules for Intro To CUDA
========================================


EOU
}

ibcm-url(){ echo http://github.com/simoncblyth/bcm.git ; }

ibcm-base(){ echo $(itc-prefix)/externals/ibcm ; }
ibcm-prefix(){ echo $(itc-prefix)/externals ; }

ibcm-dir(){  echo $(ibcm-base)/bcm ; }
ibcm-bdir(){  echo $(ibcm-base)/bcm.build ; }
ibcm-bcd(){ cd $(ibcm-bdir) ; }

ibcm-info(){ cat << EOI

   ibcm-url    : $(ibcm-url)
   ibcm-base   : $(ibcm-base)
   ibcm-prefix : $(ibcm-prefix)
   ibcm-dir    : $(ibcm-dir)

EOI
}


ibcm-get(){
   local iwd=$PWD
   local dir=$(dirname $(ibcm-dir)) &&  mkdir -p $dir && cd $dir
   if [ ! -d "bcm" ]; then 
       git clone $(ibcm-url)
   fi  
   cd $iwd
}

ibcm-cmake(){
  local iwd=$PWD
  local bdir=$(ibcm-bdir)
  mkdir -p $bdir
  ibcm-bcd
  cmake $(ibcm-dir) -DCMAKE_INSTALL_PREFIX=$(ibcm-prefix) 
  cd $iwd
}

ibcm-make()
{
  local iwd=$PWD
  ibcm-bcd
  cmake --build . --target ${1:-install}
  cd $iwd
}


ibcm--(){
  ibcm-get
  ibcm-cmake
  ibcm-make install
}




