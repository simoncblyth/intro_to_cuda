// nvcc vadd.cu -run && rm a.out 

#include <stdio.h>
#include <assert.h>


#define N 512

__global__ void add(int *a, int *b, int *c) 
{ 
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void random_ints( int* v, int num)
{
   for(int i=0 ; i < num ; i++ ) v[i] = rand() % 100 ; 
}

int main(void) 
{
    int *a, *b, *c; // 
    int *d_a, *d_b, *d_c; // device memory pointers to copies of a, b, c 

    int size = N * sizeof(int);

    // Alloc space for device copies of a, b, c 
    cudaMalloc((void **)&d_a, size); 
    cudaMalloc((void **)&d_b, size); 
    cudaMalloc((void **)&d_c, size);


    a = (int *)malloc(size); 
    b = (int *)malloc(size); 
    c = (int *)malloc(size);

    random_ints(a, N); 
    random_ints(b, N); 

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks
    add<<<N,1>>>(d_a, d_b, d_c);

   
    // Copy result back to host 
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    
    for(int i=0 ; i < N ; i++)
    {
        printf( " i %d   a %d b %d c %d  \n", i, a[i], b[i], c[i] );
        int c_expect = a[i] + b[i] ; 
        assert( c[i] == c_expect );
    }


    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;  
}



