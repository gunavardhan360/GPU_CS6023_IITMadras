/*
author : Anshul Farkya
created on : 12-feb-2021
*/
#include <stdio.h>
#include <stdlib.h>
#include "kernels.h"


// Print the matrix
void printMatrix(int *mat, int rows, int cols)
{
  for(int i=0; i < rows; i++) 
  {
    for(int j=0; j < cols; j++) 
	{ 
      printf("%d ", mat[i*cols+j]); 
	}
    printf("\n");
  } 
  printf("\n");
}

// Verify the correctness by comparing the sequential output with parallel output
bool compareMatrices(int *gpu, int *ref, int rows, int cols)
{

		for(int i=0; i < rows; i++)
		{
				for(int j=0; j < cols; j++)
				{
						if (ref[i*cols + j] != gpu[i*cols +j])
						{
								return false;
						}
				}
		}
		return true; // generated output matches expected output
}

// Generating the expected output 
void add_matrices(int A[], int B[], int out[], int rows, int cols)
{
  for(int i=0; i<rows; i++)
  {
	for(int j=0; j<cols; j++)
	{
        	out[i*cols + j] = A[i*cols + j] + B[i*cols + j];
	}
  }
}


int main(int argc, char** argv)
{
	FILE *fp;
	fp = fopen(argv[1], "r");
	if(fp == NULL)
	{
			printf("Failed to open input file %s\n", argv[1]);
			exit(0);
	}
	
	// Reading the input matrix from file
	int rows,cols; // holds the size of the input square matrix
	fscanf(fp, "%d", &rows);
	fscanf(fp, "%d", &cols);
	//printf("rows = %d\n",rows);
	//printf("cols = %d\n",cols);
	int numBytes = rows * cols * sizeof(int);
	
	long long k1_blkDimX, k1_gridDimX, k2_blkDimX, k2_blkDimY, k2_gridDimX, k3_blkDimX, k3_blkDimY, k3_gridDimX, k3_gridDimY;
	k1_blkDimX = 512;
	k1_gridDimX = ceil(float(rows*cols)/k1_blkDimX);
	//printf("k1_gridDimX = %d\n",k1_gridDimX);
	int *A = (int*) malloc(numBytes);
	for(int i = 0; i < rows*cols; i++)
	{
			fscanf(fp, "%d", &A[i]);
	}
	
	int *B = (int*) malloc(numBytes);
	for(int i = 0; i < rows*cols; i++)
	{
			fscanf(fp, "%d", &B[i]);
	}

	// computing the addition on CPU to generate expected output
	int *cpuOut = (int*) malloc(numBytes);
	add_matrices(A, B, cpuOut, rows, cols);
	//printMatrix(cpuOut, rows,cols);

	int *d_A;
	cudaMalloc(&d_A, numBytes);
	cudaMemcpy(d_A, A, numBytes, cudaMemcpyHostToDevice);
	
	int *d_B;
	cudaMalloc(&d_B, numBytes);
	cudaMemcpy(d_B, B, numBytes, cudaMemcpyHostToDevice);
	
	int *d_C;
	cudaMalloc(&d_C, numBytes);
	
	// Launching the first kernel
	per_row_kernel<<<k1_gridDimX, k1_blkDimX>>>(rows,cols,d_A, d_B, d_C);
	cudaDeviceSynchronize();
	// Copying back from the device
	int *out1 = (int*) malloc(numBytes);
	cudaMemcpy(out1, d_C, numBytes, cudaMemcpyDeviceToHost);
	//printMatrix(out1, rows,cols);
	printf("%s,%s\n", argv[1], compareMatrices(out1, cpuOut, rows, cols) ? "Success" : "Failure");
	
	cudaMemset(d_C, 0, numBytes);
	k2_blkDimX = 32;
	k2_blkDimY = 32;
	k2_gridDimX = ceil(float(rows*cols)/(k2_blkDimX*k2_blkDimY));
	// Launching the second kernel
	dim3 block2D(k2_blkDimX, k2_blkDimY,1);
	per_column_kernel<<<k2_gridDimX, block2D>>>(rows,cols,d_A, d_B, d_C);
	cudaDeviceSynchronize();
	// Copying back from the device
	int *out2 = (int*) malloc(numBytes);
	cudaMemcpy(out2, d_C, numBytes, cudaMemcpyDeviceToHost);
  //printMatrix(out2, rows,cols);
  printf("%s,%s\n", argv[1],compareMatrices(out2, cpuOut, rows, cols) ? "Success" : "Failure");

	cudaMemset(d_C, 0, numBytes);
	k3_blkDimX = 32;
	k3_blkDimY = 32;
	k3_gridDimX = 32;
	k3_gridDimY = ceil(float(rows*cols)/(k3_blkDimX*k3_blkDimY*k3_gridDimX));
	
	// Launching the third kernel
	dim3 grid2D(k3_gridDimX, k3_gridDimY);
	dim3 blk2D(k3_blkDimX,k3_blkDimY);
	per_element_kernel<<<grid2D, blk2D>>>(rows,cols,d_A, d_B, d_C);
	cudaDeviceSynchronize();
	// Copying back from the device
	int *out3 = (int*) malloc(numBytes);
	cudaMemcpy(out3, d_C, numBytes, cudaMemcpyDeviceToHost);
	printMatrix(out3, rows,cols);
	printf("%s,%s\n", argv[1],compareMatrices(out3, cpuOut, rows,cols) ? "Success" : "Failure");

	// Deallocating the memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(cpuOut);
	free(out1);
	free(out2);
	free(out3);

	return(0);

}
