itc-source(){ echo $BASH_SOURCE ; }
itc-vi(){ vi $(itc-source) ; }
itc-env(){ echo -n ; } 
itc-home(){ echo $(dirname $(itc-source)) ; }
itc-cd(){ cd $(itc-home) ; }

itc-usage(){ cat << EOU

Intro To CUDA
================

Hookup to your bash shell with:: 

  itc-(){  . $HOME/intro_to_cuda/itc.bash && itc-env ; } 


EOU
}

#itc-base(){    echo ${LOCAL_BASE:-/usr/local} ; }
itc-base(){    echo /tmp/$USER ; }
itc-prefix(){  echo $(itc-base)/intro_to_cuda ; }

itc-info(){ cat << EOI

   itc-home   : $(itc-home)
   itc-base   : $(itc-base)
   itc-prefix : $(itc-prefix)

EOI
}


itc-externals(){  cat << EOX
ibcm
EOX
}


itc-bdir()
{
   local sdir=${1:-$(pwd)}
   local bdir=$(itc-prefix)/build/$(basename $sdir)
   echo $bdir
}


itc-cmake()
{
   local sdir=$(pwd)
   local bdir=$(itc-bdir $sdir)
   local idir=$(itc-prefix)

   #rm -rf $bdir 
   mkdir -p $bdir && cd $bdir && pwd 

   cmake $sdir \
         -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_PREFIX_PATH=$(itc-prefix)/externals \
         -DCMAKE_INSTALL_PREFIX=$idir 

   cd $sdir && pwd
}

itc-make()
{
   local sdir=$(pwd)   
   local bdir=$(itc-bdir $sdir)
   cd $bdir

   make VERBOSE=1
   make install VERBOSE=1  

   cd $sdir
}


ibcm-(){  . $(itc-home)/ibcm.bash && ibcm-env ; }










