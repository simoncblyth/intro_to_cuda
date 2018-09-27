/**
demoTex
==========

Adapted from sample code on the slides 

* http://on-demand.gputechconf.com/gtc-express/2011/presentations/texture_webinar_aug_2011.pdf


**/

#include <thrust/host_vector.h>

texture<float, cudaTextureType2D, cudaReadModeElementType> tex;   // global declaration visible from host and device 

__global__ void demoTex(int width, int height)
{   
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    float x = ix/(float)width + 0.5f ;
    float y = iy/(float)height + 0.5f ;

    float v = tex2D(tex, x, y ); 

    printf("demoTex : Thread index: (%i, %i); v = %f\n", ix, iy, v ); 
}


int main()
{
    int width = 3 ; 
    int height = 3;
    thrust::host_vector<float> h_data(width*height, 3.f);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    int size = h_data.size()*sizeof(float) ; 

    void* h_data_ptr = thrust::raw_pointer_cast(h_data.data()) ;
   
    cudaMemcpyToArray(cuArray, 0, 0, h_data_ptr, size, cudaMemcpyHostToDevice);

    tex.addressMode[0] = cudaAddressModeWrap; 
    tex.addressMode[1] = cudaAddressModeWrap; 
    tex.filterMode = cudaFilterModeLinear; // "Filter" means interpolation
    tex.normalized = true; 

    cudaBindTextureToArray(tex, cuArray, channelDesc); 

    dim3 grid(1,1) ; 
    dim3 block(3,3); 

    demoTex<<<grid, block>>>(width, height);    


    cudaDeviceSynchronize();  // without Synchronize the process terminates before printf output appears 

    return 0 ; 
}


