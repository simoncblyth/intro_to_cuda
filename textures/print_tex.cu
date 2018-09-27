//  https://stackoverflow.com/questions/14927524/read-cudaarray-in-device-code
//  sample code from JackOLantern


#include <stdio.h>
#include <thrust/device_vector.h>

// --- 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// --- 2D surface memory
surface<void, 2> surf2D;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*************************************/
/* cudaArray PRINTOUT TEXTURE KERNEL */
/*************************************/
__global__ void cudaArrayPrintoutTexture(int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    printf("Thread index: (%i, %i); cudaArray = %f\n", x, y, tex2D(texRef, x / (float)width + 0.5f, y / (float)height + 0.5f));
}

/*************************************/
/* cudaArray PRINTOUT TEXTURE KERNEL */
/*************************************/
__global__ void cudaArrayPrintoutSurface(int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float temp;

    surf2Dread(&temp, surf2D, x * 4, y);

    printf("Thread index: (%i, %i); cudaArray = %f\n", x, y, temp);
}

/********/
/* MAIN */
/********/
int main()
{
    int width = 3, height = 3;

    thrust::host_vector<float> h_data(width*height, 3.f);

    // --- Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaArray* cuArray;

    /*******************/
    /* TEXTURE BINDING */
    /*******************/
    gpuErrchk(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // --- Copy to host data to device memory
    gpuErrchk(cudaMemcpyToArray(cuArray, 0, 0, thrust::raw_pointer_cast(h_data.data()), width*height*sizeof(float), cudaMemcpyHostToDevice));

    // --- Set texture parameters
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    texRef.filterMode = cudaFilterModeLinear;
    texRef.normalized = true;

    // --- Bind the array to the texture reference
    gpuErrchk(cudaBindTextureToArray(texRef, cuArray, channelDesc));

    // --- Invoking printout kernel
    dim3 dimBlock(3, 3);
    dim3 dimGrid(1, 1);
    cudaArrayPrintoutTexture<<<dimGrid, dimBlock>>>(width, height);

    gpuErrchk(cudaUnbindTexture(texRef));

    gpuErrchk(cudaFreeArray(cuArray));

    /******************/
    /* SURFACE MEMORY */
    /******************/
    gpuErrchk(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore));

    // --- Copy to host data to device memory
    gpuErrchk(cudaMemcpyToArray(cuArray, 0, 0, thrust::raw_pointer_cast(h_data.data()), width*height*sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaBindSurfaceToArray(surf2D, cuArray));

    cudaArrayPrintoutSurface<<<dimGrid, dimBlock>>>(width, height);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFreeArray(cuArray));

    return 0 ; 
}
