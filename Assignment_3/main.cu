#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include <sys/time.h>
using namespace std;

__global__ void time_at_each_tolltax_points(int n, int x, float *time, int* sort, float* taxlocal) {
	int id = threadIdx.x;
    int m = blockDim.x;
	taxlocal[id] = 0;
	for(int i = id; i < n; i += m){
		if(time[sort[i]] > taxlocal[id]) time[sort[i]] += x;
		else time[sort[i]] = taxlocal[id] + x;
        taxlocal[id] = time[sort[i]];
	}
}

__global__ void time_per_each_tollzone(int k, int n, int dis, float *time, int *speed) {
	int id = blockIdx.x;
	time[id] += float(dis)/speed[k*n+id] * 60;
}

void numbering(int n, float *temp, int *sort){
	int min;
	for(int i = 0; i < n; i++){
		min = 0;
		for(int j = 1; j < n; j++) if(temp[min] > temp[j]) min = j;
		temp[min] = INT_MAX;
		sort[i] = min;
	}
}

//Complete the following function
void operations ( int n, int k, int m, int x, int dis, int *hspeed, int **results )  {
    float *time, *htime, *temp, *taxlocal;
	int *speed, *hsort, *sort;
    cudaMalloc(&speed, n*( k+1 ) * sizeof (int) );
    cudaMalloc(&time, n*sizeof(float));
    cudaMalloc(&sort, n*sizeof(int));
    cudaMalloc(&taxlocal, m*sizeof(float));

    htime = (float *)malloc(n*sizeof(float));
    temp = (float *)malloc(n*sizeof(float));
    hsort = (int *)malloc(n*sizeof(int));
	for(int i = 0; i < n; i++) htime[i] = 0;

    cudaMemcpy(time, htime, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(speed, hspeed, n*( k+1 ) * sizeof (int) , cudaMemcpyHostToDevice);

	for(int a = 0; a < k; a++){
		time_per_each_tollzone<<<n, 1>>>(a, n, dis, time, speed);
        cudaMemcpy(temp, time, n*sizeof(float), cudaMemcpyDeviceToHost);
        numbering(n, temp, hsort);
        results[0][a] = hsort[0] + 1;
        results[1][a] = hsort[n-1] + 1;
        cudaMemcpy(sort, hsort, n*sizeof(int), cudaMemcpyHostToDevice);
        time_at_each_tolltax_points<<<1, m>>>(n, x, time, sort, taxlocal);
	}
    time_per_each_tollzone<<<n, 1>>>(k, n, dis, time, speed);
    cudaMemcpy(htime, time, n*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++) results[2][i] = (int)htime[i]; 
    numbering(n, htime, hsort);
    results[0][k] = hsort[0] + 1;
    results[1][k] = hsort[n-1] + 1;
}

int main(int argc,char **argv){

    //variable declarations
    int n,k,m,x;
    int dis;
    
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "textcase1.txt file failed to open." );
        return 0;
    }
    
    
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of vehicles
    fscanf( inputfilepointer, "%d", &k );      //scaning for number of toll tax zones
    fscanf( inputfilepointer, "%d", &m );      //scaning for number of toll tax points
    fscanf( inputfilepointer, "%d", &x );      //scaning for toll tax zone passing time
    
    fscanf( inputfilepointer, "%d", &dis );    //scaning for distance between two consecutive toll tax zones


    // scanning for speeds of each vehicles for every subsequent toll tax combinations
    int *speed = (int *) malloc ( n*( k+1 ) * sizeof (int) );
    for ( int i=0; i<=k; i++ )  {
        for ( int j=0; j<n; j++ )  {
            fscanf( inputfilepointer, "%d", &speed[i*n+j] );
        }
    }
    
    // results is in the format of first crossing vehicles list, last crossing vehicles list 
    //               and total time taken by each vehicles to pass the highway
    int **results = (int **) malloc ( 3 * sizeof (int *) );
    results[0] = (int *) malloc ( (k+1) * sizeof (int) );
    results[1] = (int *) malloc ( (k+1) * sizeof (int) );
    results[2] = (int *) malloc ( (n) * sizeof (int) );


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);


    // Function given to implement
    operations ( n, k, m, x, dis, speed, results );


    cudaDeviceSynchronize();

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    // First crossing vehicles list
    for ( int i=0; i<=k; i++ )  {
        fprintf( outputfilepointer, "%d ", results[0][i]);
    }
    fprintf( outputfilepointer, "\n");


    //Last crossing vehicles list
    for ( int i=0; i<=k; i++ )  {
        fprintf( outputfilepointer, "%d ", results[1][i]);
    }
    fprintf( outputfilepointer, "\n");


    //Total time taken by each vehicles to pass the highway
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", results[2][i]);
    }
    fprintf( outputfilepointer, "\n");

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    return 0;
}