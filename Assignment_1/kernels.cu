// Code by CH18B035 : AKiti Gunavardhan Reddy
#include <stdio.h>
#include <cuda.h>

__global__ void per_row_kernel(int m, int n, int *A, int *B, int *C){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < m)
        for(int i = 0; i < n; i++) C[n*id + i] = A[n*id + i] + B[n*id + i];
}

__global__ void per_column_kernel(int m,int n,int *A,int *B,int *C){
    unsigned id = blockIdx.x*(blockDim.x*blockDim.y)  + threadIdx.x * blockDim.y + threadIdx.y;
    if(id < n)
        for(int i = 0; i < m; i++) C[n*i + id] = A[n*i + id] + B[n*i + id];
}

__global__ void per_element_kernel(int m,int n,int *A,int *B,int *C){
    unsigned id =  (blockIdx.x * gridDim.y + blockIdx.y)*(blockDim.x*blockDim.y)  + threadIdx.x * blockDim.y + threadIdx.y;
    if(id < m*n) C[id] = A[id] + B[id];
}
