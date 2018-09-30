iminuit2-source(){   echo ${BASH_SOURCE} ; }
iminuit2-edir(){ echo $(dirname $(iminuit2-source)) ; }
iminuit2-ecd(){  cd $(iminuit2-edir); }

iminuit2-base(){  echo $(itc-prefix)/externals/iminuit2 ; }
iminuit2-prefix(){ echo $(itc-prefix)/externals ; }
iminuit2-dir(){  echo $(iminuit2-base)/Minuit2 ; }

iminuit2-cd(){   cd $(iminuit2-dir); }
iminuit2-vi(){   vi $(iminuit2-source) ; }
iminuit2-env(){  echo -n ; }
iminuit2-usage(){ cat << EOU

Minuit2 
=========

* https://root.cern.ch/doc/v608/Minuit2Page.html

A standalone version of Minuit2 (independent of ROOT) can be downloaded from

* https://root.cern.ch/doc/Minuit2.tar.gz  

  * BUT THATS A BROKEN LINK 

It does not contain the ROOT interface and it is therefore totally
independent of external packages and can be simply build using the configure
script and then make. Example tests are provided in the directory test/MnSim
and test/MnTutorial and they can be built with the make check command. The
Minuit2 User Guide provides all the information needed for using directly
(without add-on packages like ROOT) Minuit2.

* https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.html

* https://github.com/jpivarski/pyminuit2

* http://seal.web.cern.ch/seal/work-packages/mathlibs/minuit/release/download.html


INSTEAD GET THIS ONE

5.34.14	
2014/01/24

* http://www.cern.ch/mathlibs/sw/5_34_14/Minuit2/Minuit2-5.34.14.tar.gz



EOU
}

#iminuit2-url(){ echo https://root.cern.ch/doc/Minuit2.tar.gz ; }   BROKEN
iminuit2-url(){ echo http://www.cern.ch/mathlibs/sw/5_34_14/Minuit2/Minuit2-5.34.14.tar.gz ; }

iminuit2-info(){ cat << EOI

   iminuit2-url    : $(iminuit2-url)
   iminuit2-prefix : $(iminuit2-prefix)


EOI
}


iminuit2-get(){
   local dir=$(dirname $(iminuit2-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(iminuit2-url)
   local dst=$(basename $url) 
   local nam=$(echo ${dst/.tar.gz})

   [ ! -f "$dst" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $dst 
   ln -svf $nam Minuit2 
}

iminuit2-build()
{
   iminuit2-cd
   ./configure --prefix=$(iminuit2-prefix)
   make
   make install
}

iminuit2-build-notes(){ cat << EON
----------------------------------------------------------------------
Libraries have been installed in:

    /tmp/blyth/intro_to_cuda/externals/lib

EON
}

iminuit2--()
{
   iminuit2-get
   iminuit2-build
}

iminuit2-check()
{
   iminuit2-cd
   make check  
   ./test/MnTutorial/test_Minuit2_Quad4FMain
}

iminuit2-doc(){ open https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.html ; }


