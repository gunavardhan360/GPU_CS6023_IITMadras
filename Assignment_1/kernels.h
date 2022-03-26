#ifndef _KERNELS_H_
#define _KERNELS_H_

__global__ void per_row_kernel(int m,int n,int *A,int *B,int *C);

__global__ void per_column_kernel(int m,int n,int *A,int *B,int *C);

__global__ void per_element_kernel(int m,int n,int *A,int *B,int *C);

#endif
