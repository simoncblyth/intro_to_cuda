/**

nvcc demoTex2.cu && ./a.out && rm a.out 

demoTex2
==========

Adapted from sample code on the slides 

* http://on-demand.gputechconf.com/gtc-express/2011/presentations/texture_webinar_aug_2011.pdf

**/

#include <cassert>
#include <iostream>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>


texture<float, cudaTextureType2D, cudaReadModeElementType> tex;   // global declaration visible from host and device 

__global__ void demoTexKernel(int nx, int ny, float* d_out)
{   
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned idx = nx*iy + ix ; 

    float x = (float(ix) + 0.5f)/(float)nx ;
    float y = (float(iy) + 0.5f)/(float)ny ;

    float v = tex2D(tex, x, y );
    d_out[idx] = v ; 

    printf("demoTexKernel : idx %i blockDim (%i,%i) blockIdx (%i,%i) threadIdx (%i,%i) (ix,iy): (%i, %i); v = %f\n", idx, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, ix, iy, v ); 
}


struct demoTex 
{
    int nx ; 
    int ny ; 

    cudaChannelFormatDesc channelDesc ; 
    cudaArray* cuArray;

    demoTex(int nx_, int ny_ ) : nx(nx_), ny(ny_) 
    {
        initArray();
        initTex();
    }
    int total() const { return nx*ny ; } 

    void initArray()
    {
        channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cuArray, &channelDesc, nx, ny);
    }
    void initTex()
    {    
        tex.addressMode[0] = cudaAddressModeWrap; 
        tex.addressMode[1] = cudaAddressModeWrap; 
        tex.filterMode = cudaFilterModeLinear; // "Filter" means interpolation
        tex.normalized = true ; 
        cudaBindTextureToArray(tex, cuArray, channelDesc); 
    }

    void demoFillHost(float* h_buf)
    {
        for( int ix = 0 ; ix < nx ; ix++ )
        {
            for( int iy = 0 ; iy < ny ; iy++ )
            {
                int index = ix*ny + iy ; 
                //float fval = float(index) ;  
                float fval = float(ix)*100+float(iy) ;  

                *(h_buf + index) = fval  ;  
            }
        } 
    }

    void fillArray(float* h_buf, int num_bytes)
    {
        cudaMemcpyToArray(cuArray, 0, 0, h_buf, num_bytes, cudaMemcpyHostToDevice);
    }

    void launch(float* d_out)
    {
        dim3 grid(1,1) ; 
        dim3 block(nx,ny); 

        demoTexKernel<<<grid, block>>>(nx, ny, d_out);    
    }

};


template <typename Vector>
void print(const char* s, const Vector& v)
{
  typedef typename Vector::value_type T;

  std::cout << s << " : " ;
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}



int main()
{
    //demoTex dt(4,4); 
    //demoTex dt(16,16); 
    demoTex dt(10,4); 

    thrust::host_vector<float> h_in(dt.total());
    //thrust::sequence( h_in.begin(), h_in.end() ) ;  

    float* h_in_ptr = thrust::raw_pointer_cast(h_in.data()) ;
    dt.demoFillHost( h_in_ptr ); 
    print( "h_in", h_in ); 

    int num_bytes = h_in.size()*sizeof(float) ; 

    dt.fillArray( h_in_ptr, num_bytes );  

    thrust::device_vector<float> d_out(dt.total());
    float* d_out_ptr = thrust::raw_pointer_cast(d_out.data()) ; 

    dt.launch(d_out_ptr); 

    thrust::host_vector<float> h_out(d_out);
    print( "h_out", h_out ); 

    thrust::minus<float>  bop_minus ;
    thrust::maximum<float> bop_maximum ; 

    float maxdiff = thrust::inner_product(h_in.begin(), h_in.end(), h_out.begin(), 0.f , bop_maximum, bop_minus );
    std::cout << " maxdiff " << maxdiff << std::endl ; 
    assert( maxdiff == 0.f ); 


    cudaDeviceSynchronize();  // without Synchronize the process terminates before printf output appears 

    return 0 ; 
}


