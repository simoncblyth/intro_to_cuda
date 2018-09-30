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

itc-banner()
{
   local msg=${1:-$FUNCNAME}
   local sdir=$(pwd)
   local bdir=$(itc-bdir $sdir)
   printf "%20s : sdir %-50s        bdir %-50s  \n" $msg $sdir $bdir
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

itc-bwipe()
{
   itc-banner $FUNCNAME
   local sdir=$(pwd)
   local bdir=$(itc-bdir $sdir)
   rm -rf $bdir   
}

itc-cmake()
{
   itc-banner $FUNCNAME
   local sdir=$(pwd)
   local bdir=$(itc-bdir $sdir)
   local idir=$(itc-prefix)

   #rm -rf $bdir 
   mkdir -p $bdir && cd $bdir && pwd 

   cmake $sdir \
         -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_PREFIX_PATH=$(itc-prefix)/externals \
         -DCMAKE_INSTALL_PREFIX=$idir \
         -DCMAKE_MODULE_PATH=$(itc-home)/cmake/Modules 

   cd $sdir && pwd
}

itc-make()
{
   itc-banner $FUNCNAME
   local sdir=$(pwd)   
   local bdir=$(itc-bdir $sdir)
   cd $bdir

   make VERBOSE=1
   make install VERBOSE=1  

   cd $sdir
}


# installers for externals 
ibcm-(){      . $(itc-home)/ibcm.bash && ibcm-env ; }
iminuit2-(){  . $(itc-home)/iminuit2.bash && iminuit2-env ; }










