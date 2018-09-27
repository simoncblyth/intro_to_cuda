Textures 
===========

* "cudaFilterMode tex2D texture thrust"



References
-------------

* http://on-demand.gputechconf.com/gtc-express/2011/presentations/texture_webinar_aug_2011.pdf


Opticks texture tests using OptiX
-----------------------------------

The below are "standalone" tests, within OptiX context, that 
sample across entire textures, writing output to buffer and 
checking get expected results. 

* ~/opticks/optixrap/tests/OOtex0Test.cc
* ~/opticks/optixrap/cu/tex0Test.cc

* ~/opticks/optixrap/tests/OOtexTest.cc
* ~/opticks/optixrap/cu/texTest.cc

* TODO: try to reimplement using Thrust+CUDA alone to serve as an example of textures with Thrust 




Samples
---------

::

    epsilon:samples blyth$ cuda-
    epsilon:samples blyth$ cuda-samples-tex2D
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/SobelFilter/SobelFilter_kernels.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/postProcessGL/postProcessGL.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/boxFilter/boxFilter_kernel.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/HSOpticalFlow/warpingKernel.cuh
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/HSOpticalFlow/derivativesKernel.cuh
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/HSOpticalFlow/upscaleKernel.cuh
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/HSOpticalFlow/downscaleKernel.cuh
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/bilateralFilter/bilateral_kernel.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/convolutionTexture/convolutionTexture.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/dct8x8/dct8x8_kernel1.cuh
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/stereoDisparity/stereoDisparity_kernel.cuh
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/stereoDisparity/stereoDisparity
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/stereoDisparity/stereoDisparity.o
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/stereoDisparity/stereoDisparity.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/imageDenoising/imageDenoising_nlm_kernel.cuh
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/imageDenoising/imageDenoising_nlm2_kernel.cuh
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/imageDenoising/imageDenoising_knn_kernel.cuh
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/imageDenoising/imageDenoising_copy_kernel.cuh
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/3_Imaging/bicubicTexture/bicubicTexture_kernel.cuh
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/bin/x86_64/darwin/release/stereoDisparity
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/2_Graphics/volumeFiltering/volumeRender_kernel.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/2_Graphics/bindlessTexture/bindlessTexture_kernel.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/6_Advanced/FunctionPointers/FunctionPointers_kernels.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/6_Advanced/lineOfSight/lineOfSight.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/0_Simple/simplePitchLinearTexture/simplePitchLinearTexture.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/0_Simple/simpleTexture/simpleTexture.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/0_Simple/simpleTextureDrv/simpleTexture_kernel.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/0_Simple/simpleLayeredTexture/simpleLayeredTexture.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/0_Simple/simpleSurfaceWrite/simpleSurfaceWrite.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/5_Simulations/fluidsGL/fluidsGL_kernels.cu
    epsilon:samples blyth$ 


    epsilon:samples blyth$ cuda-samples-tex3D
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/2_Graphics/simpleTexture3D/simpleTexture3D_kernel.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/2_Graphics/volumeFiltering/volumeFilter_kernel.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/2_Graphics/volumeFiltering/volumeRender_kernel.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/2_Graphics/volumeRender/volumeRender_kernel.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/5_Simulations/smokeParticles/particles_kernel_device.cuh
    epsilon:samples blyth$ 

