rotate_image using CUDA "bindless" Texture Objects 
======================================================

1. PNG image file is loaded and copied into a GPU texture.  
2. CUDA kernel is launched to rotate the image by an input angle, reading from the texture
3. output from GPU global memory is downloaded and written to a PNG output file

This is based on 

* /usr/local/cuda/samples/0_Simple/simpleTexture/simpleTexture.cu  

but its using standalone image handling from 

* stb_image.h stb_image_write.h and SIMG.hh  (these are all from opticks/sysrap) 


Usage::

    ./SIMGStandaloneTest.sh /path/to/input.png /path/to/output.png 

The input image must be 4-channel RGBA, as textures are fussy about byte alignment.


The textures are created using the below CUDA functions and types::

    cudaTextureDesc 
    cudaTextureObject_t 
    cudaCreateTextureObject 
    cudaDestroyTextureObject

Note that *Texture Objects* are also called "bindless" textures, as they are 
more convenient that the old static ones and can be referenced in kernel 
argument lists.

*Texture Objects* were introduced long ago with Kepler GPUs and CUDA 5.0 
a blog post introduing those is linked below.

* https://developer.nvidia.com/blog/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/


They are much more convenient to use than the legacy global context approach to texture.


Other references in textures:

* http://on-demand.gputechconf.com/gtc-express/2011/presentations/texture_webinar_aug_2011.pdf

* https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory




This examples was initially developed in an Opticks/sysrap test:: 

   ~/opticks/sysrap/tests/SIMGStandaloneTest.sh  
   ~/opticks/sysrap/tests/SIMGStandaloneTest.cc









