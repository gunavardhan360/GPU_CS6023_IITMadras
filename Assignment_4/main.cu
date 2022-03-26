// Code by Gunavardhan CH18B035
#include <cstdlib>
#include<stdio.h>
#include<stdlib.h>
#include <stdbool.h>
#include<cuda.h>
#include <sys/time.h>
#include <string.h>
#define order 8
using namespace std;

typedef struct records {
    int* value;
    int lockvar;
} records;
  
  // Node
typedef struct node {
    void **pointers;
    int *keys;
    struct node *parent;
    bool is_leaf;
    int len_keys;
    struct node *next;
} node;

 __device__ __host__ node *findLeaf(node *const root, int key);
 __device__ __host__ records *find(node *root, int key, node **leaf_out);
 node *insertIntoLeaf(node *leaf, int key, records *pointer);
 node *insertIntoLeafAfterSplitting(node *root, node *leaf, int key, records *pointer);
 node *insertIntoNode(node *root, node *parent, int left_index, int key, node *right);
 node *insertIntoNodeAfterSplitting(node *root, node *parent, int left_index, int key, node *right);
 node *insertIntoParent(node *root, node *left, int key, node *right);
 node *insertIntoNewRoot(node *left, int key, node *right);
 node *newbptree(int key, records *pointer);
 node *insertbptree(node *root, int key, int* value);

__device__ int findRange(node *const root, int key_start, int key_end, int returned_keys[], void *returned_pointers[]) {
    int i, num_found;
    num_found = 0;
    node *n = findLeaf(root, key_start);
    if (n == NULL) return 0;
    for (i = 0; i < n->len_keys && n->keys[i] < key_start; i++);
    while (n != NULL) {
        for (; i < n->len_keys && n->keys[i] <= key_end; i++){
            returned_keys[num_found] = n->keys[i];
            returned_pointers[num_found] = n->pointers[i];
            num_found++;
        }
        n = (node *)n->pointers[order - 1];
        i = 0;
    }
    return num_found;
}

__device__ __host__ node *findLeaf(node *const root, int key) {
    int i = 0;
    node *c = root;
    while (!c->is_leaf) {
        i = 0;
        while (i < c->len_keys) {
            if (key >= c->keys[i])
                i++;
            else
                break;
        }   
        c = (node *)c->pointers[i];
    }
    return c;
}

__device__ __host__ records *find(node *root, int key, node **leaf_out) {
    if (root == NULL) {
        if (leaf_out != NULL) {
            *leaf_out = NULL;
        }
        return NULL;
    }
  
    int i = 0;
    node *leaf = NULL;
  
    leaf = findLeaf(root, key);
  
    for (i = 0; i < leaf->len_keys; i++)
        if (leaf->keys[i] == key) break;
    if (leaf_out != NULL) {
        *leaf_out = leaf;
    }
    if (i == leaf->len_keys)
        return NULL;
    else
        return (records *)leaf->pointers[i];
}

 records *newRecord(int* value) {
    records *new_record = (records *)malloc(sizeof(records));
    new_record->value = value;
    new_record->lockvar = 0;
    return new_record;
}

 node *newNode(void) {
    node *new_node;
    new_node = (node *)malloc(sizeof(node));
    new_node->keys = (int *)malloc((order - 1) * sizeof(int));
    new_node->pointers = (void **)malloc(order * sizeof(void *));
    new_node->is_leaf = false;
    new_node->len_keys = 0;
    new_node->parent = NULL;
    new_node->next = NULL;
    return new_node;
}
  
 node *newLeaf(void) {
    node *leaf = newNode();
    leaf->is_leaf = true;
    return leaf;
}

 node *insertIntoNewRoot(node *left, int key, node *right) {
    node *root = newNode();
    root->keys[0] = key;
    root->pointers[0] = left;
    root->pointers[1] = right;
    root->len_keys++;
    root->parent = NULL;
    left->parent = root;
    right->parent = root;
    return root;
}

 node *insertIntoParent(node *root, node *left, int key, node *right) {
    int left_index;
    node *parent;
  
    parent = left->parent;
  
    if (parent == NULL)
        return insertIntoNewRoot(left, key, right);
  
    left_index = 0;
    while (left_index <= parent->len_keys && parent->pointers[left_index] != left) left_index++;
  
    if (parent->len_keys < order - 1)
        return insertIntoNode(root, parent, left_index, key, right);
  
    return insertIntoNodeAfterSplitting(root, parent, left_index, key, right);
}

 node *insertIntoNodeAfterSplitting(node *root, node *old_node, int left_index, int key, node *right) {
    int i, j, split, k_prime;
    node *new_node, *child;
    int *temp_keys;
    node **temp_pointers;

    temp_pointers = (node **)malloc((order + 1) * sizeof(node *));
    
    temp_keys = (int *)malloc(order * sizeof(int));

    for (i = 0, j = 0; i < old_node->len_keys + 1; i++, j++) {
        if (j == left_index + 1) j++;
        temp_pointers[j] = (node *)old_node->pointers[i];
    }

    for (i = 0, j = 0; i < old_node->len_keys; i++, j++) {
        if (j == left_index) j++;
        temp_keys[j] = old_node->keys[i];
    }

    temp_pointers[left_index + 1] = right;
    temp_keys[left_index] = key;

    if((order-1) % 2 == 0) split = (order+1)/2;
    else split = (order+1)/2 + 1;

    new_node = newNode();
    old_node->len_keys = 0;
    for (i = 0; i < split - 1; i++) {
        old_node->pointers[i] = temp_pointers[i];
        old_node->keys[i] = temp_keys[i];
        old_node->len_keys++;
    }
    old_node->pointers[i] = temp_pointers[i];
    k_prime = temp_keys[split - 1];
    for (++i, j = 0; i < order; i++, j++) {
        new_node->pointers[j] = temp_pointers[i];
        new_node->keys[j] = temp_keys[i];
        new_node->len_keys++;
    }
    new_node->pointers[j] = temp_pointers[i];
    free(temp_pointers);
    free(temp_keys);
    new_node->parent = old_node->parent;
    for (i = 0; i <= new_node->len_keys; i++) {
        child = (node *)new_node->pointers[i];
        child->parent = new_node;
    }

    return insertIntoParent(root, old_node, k_prime, new_node);
}

 node *insertIntoNode(node *root, node *n, int left_index, int key, node *right) {
    int i;
    for (i = n->len_keys; i > left_index; i--) {
        n->pointers[i + 1] = n->pointers[i];
        n->keys[i] = n->keys[i - 1];
    }
    n->pointers[left_index + 1] = right;
    n->keys[left_index] = key;
    n->len_keys++;
    return root;
}

 node *insertIntoLeafAfterSplitting(node *root, node *leaf, int key, records *pointer) {
    node *new_leaf;
    int *temp_keys;
    void **temp_pointers;
    int insertion_index, split, new_key, i, j;
  
    new_leaf = newLeaf();
  
    temp_keys = (int *)malloc(order * sizeof(int));
  
    temp_pointers = (void **)malloc(order * sizeof(void *));
  
    insertion_index = 0;
    while (insertion_index < order - 1 && leaf->keys[insertion_index] < key) insertion_index++;
  
    for (i = 0, j = 0; i < leaf->len_keys; i++, j++) {
        if (j == insertion_index) j++;
        temp_keys[j] = leaf->keys[i];
        temp_pointers[j] = leaf->pointers[i];
    }
  
    temp_keys[insertion_index] = key;
    temp_pointers[insertion_index] = pointer;
  
    leaf->len_keys = 0;
  
    if((order-1) % 2 == 0) split = (order-1)/2;
    else split = (order-1)/2 + 1;
  
    for (i = 0; i < split; i++) {
        leaf->pointers[i] = temp_pointers[i];
        leaf->keys[i] = temp_keys[i];
        leaf->len_keys++;
    }
  
    for (i = split, j = 0; i < order; i++, j++) {
        new_leaf->pointers[j] = temp_pointers[i];
        new_leaf->keys[j] = temp_keys[i];
        new_leaf->len_keys++;
    }
  
    free(temp_pointers);
    free(temp_keys);
  
    new_leaf->pointers[order - 1] = leaf->pointers[order - 1];
    leaf->pointers[order - 1] = new_leaf;
  
    for (i = leaf->len_keys; i < order - 1; i++) leaf->pointers[i] = NULL;
    for (i = new_leaf->len_keys; i < order - 1; i++) new_leaf->pointers[i] = NULL;
  
    new_leaf->parent = leaf->parent;
    new_key = new_leaf->keys[0];
  
    return insertIntoParent(root, leaf, new_key, new_leaf);
}

 node *insertIntoLeaf(node *leaf, int key, records *pointer) {
    int i, insertion_point;
  
    insertion_point = 0;
    while (insertion_point < leaf->len_keys && leaf->keys[insertion_point] < key) insertion_point++;
  
    for (i = leaf->len_keys; i > insertion_point; i--) {
        leaf->keys[i] = leaf->keys[i - 1];
        leaf->pointers[i] = leaf->pointers[i - 1];
    }
    leaf->keys[insertion_point] = key;
    leaf->pointers[insertion_point] = pointer;
    leaf->len_keys++;
    return leaf;
}

 node *newbptree(int key, records *pointer) {
    node *root = newLeaf();
    root->keys[0] = key;
    root->pointers[0] = pointer;
    root->pointers[order - 1] = NULL;
    root->parent = NULL;
    root->len_keys++;
    return root;
}

 node *insertbptree(node *root, int key, int* value) {
    records *records_ptr = NULL;
    node *leaf = NULL;
  
    records_ptr = find(root, key, NULL);
    if (records_ptr != NULL) {
        records_ptr->value = value;
        records_ptr->lockvar = 0;
        return root;
    }
  
    records_ptr = newRecord(value);
  
    if (root == NULL){
        return newbptree(key, records_ptr);
    }
  
    leaf = findLeaf(root, key);
  
    if (leaf->len_keys < order - 1) {
      leaf = insertIntoLeaf(leaf, key, records_ptr);
      return root;
    }
  
    return insertIntoLeafAfterSplitting(root, leaf, key, records_ptr);
}

__global__ void heightkernel(node *root, int *temph){
    int h = 0;
    node *c = root;
    while (!c->is_leaf) {
        c = (node *)c->pointers[0];
        h++;
    }
    temph[0] = h;
}

__global__ void searchkernel(node *root, int *searchq, int m, int *results){
    int idx = blockIdx.x;
    records *r = find(root, searchq[idx+2], NULL);
    if (r != NULL){
        for(int i = 0; i < m; i++) results[idx*m+i] = r->value[i];
    }
    else
        results[idx*m] = -1;
}

__global__ void rangekernel(node *root, int *rangeq, int n, int m, int *found, int *results){
    int idx = blockIdx.x;
    int id = idx*2 + 2;
    int key_start = rangeq[id], key_end = rangeq[id+1];
    int array_size = min(n, key_end - key_start + 1);
    int *returned_keys = (int *)malloc(array_size*sizeof(int));
    void **returned_ptrs = (void **)malloc(array_size*sizeof(void *));
    int temp = findRange(root, key_start, key_end, returned_keys, returned_ptrs);
    if (!temp){
        results[found[blockIdx.x]*m] = -1;
    }
    else {
        for(int j = 0; j < temp; j++){
            records *r = (records *)returned_ptrs[j];
            for(int i = 0; i < m; i++){
                results[found[blockIdx.x]*m+j*m+i] = r->value[i];
            }
        }
    }
}

__global__ void rangekernelcount(node *root, int *rangeq, int m, int *found){
    int idx = blockIdx.x;
    int id = idx*2 + 2;
    int key_start = rangeq[id], key_end = rangeq[id+1];
    int i, num_found = 0;
    node *n = findLeaf(root, key_start);
    for (i = 0; i < n->len_keys && n->keys[i] < key_start; i++);
    while (n != NULL) {
        for (; i < n->len_keys && n->keys[i] <= key_end; i++) num_found++;
        n = (node *)n->pointers[order - 1];
        i = 0;
    }
    found[blockIdx.x+1] = num_found+1;
}

__global__ void additionkernel(node *root, int *addq, int m){
    int idx = blockIdx.x;
    int id = idx*3 + 2, old;
    records *r = find(root, addq[id], NULL);
    if(r != NULL){
        do{
            old = atomicCAS(&(r->lockvar), 0, 1);
            if(old == 0){
                r->value[addq[id+1]-1] += addq[id+2];
                r->lockvar = 0;
            }
        }while(old != 0);
    }
}

__global__ void pathkernel(node *root, int key, int* results){
    int i = 0, length = 0;
    node *c = root;
    while (!c->is_leaf) {
        results[length++] = c->keys[0];
        i = 0;
        while (i < c->len_keys) {
            if (key >= c->keys[i]){
                i++;
            }
            else
                break;
        }  
        c = (node *)c->pointers[i];
    }
    results[length++] = c->keys[0];
    results[length++] = -1;
}

records *copyRecord(records *src, int m){
	records *dis, *temp = (records *)malloc(sizeof(records));
	cudaMalloc(&dis, sizeof(records));
	int *tempval;
	cudaMalloc(&tempval, m*sizeof(int));
	cudaMemcpy(tempval, src->value, m*sizeof(int), cudaMemcpyHostToDevice);
	temp->value = tempval;
	temp->lockvar = 0;
	cudaMemcpy(dis, temp, sizeof(records), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
	return dis;
}

node *copyleafnode(node *src, int m){
    node *dis, *temp = (node *)malloc(sizeof(node));
	cudaMalloc(&dis, sizeof(node));
    records **temppointers;
    cudaMalloc(&temppointers, order*sizeof(records *));
    records **tempptrs = (records **)malloc(order * sizeof(records *));
    int *tempkeys;
    cudaMalloc(&tempkeys, (order - 1) * sizeof(int));
    for(int i = 0; i < src->len_keys; i++){
        cudaMalloc(&tempptrs[i], sizeof(records));
        records *gpu = copyRecord((records *)src->pointers[i], m);
        cudaMemcpy(tempptrs[i], gpu, sizeof(records), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(temppointers, tempptrs, order*sizeof(records *), cudaMemcpyHostToDevice);
    cudaMemcpy(tempkeys, src->keys, (order - 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    temp->pointers = (void **)temppointers;
    temp->keys = tempkeys;
    temp->parent = NULL;
    temp->is_leaf = src->is_leaf;   
    temp->len_keys = src->len_keys;
    temp->next = NULL;
	cudaMemcpy(dis, temp, sizeof(node), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return dis;
}

node *copybptree(node *src, int m){
    if(src == NULL) return src;
    
    if(src->is_leaf){
        node *ans = copyleafnode(src, m); 
        cudaDeviceSynchronize();
        return ans; 
    }
    else{
        node *dis, *temp = (node *)malloc(sizeof(node));
        cudaMalloc(&dis, sizeof(node));
        node **temppointers;
        cudaMalloc(&temppointers, order*sizeof(node *));
        node **tempptrs = (node **)malloc(order * sizeof(node *));
        int *tempkeys;
        cudaMalloc(&tempkeys, (order - 1) * sizeof(int));
        for(int i = 0; i < src->len_keys + 1; i++){
            cudaMalloc(&tempptrs[i], sizeof(node));
            node *gpu = copybptree((node *)src->pointers[i], m);
            cudaMemcpy(tempptrs[i], gpu, sizeof(node), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(temppointers, tempptrs, order*sizeof(node *), cudaMemcpyHostToDevice);
        cudaMemcpy(tempkeys, src->keys, (order - 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        temp->pointers = (void **)temppointers;
        temp->keys = tempkeys;
        temp->parent = NULL;
        temp->is_leaf = src->is_leaf;
        temp->len_keys = src->len_keys;
        temp->next = NULL;
        cudaMemcpy(dis, temp, sizeof(node), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        return dis;
    }
}

__device__ node* linking(node *root, node *glbptr){
    if(root == NULL){
        printf("empty tree");
        return NULL;
    }

    if(!root->is_leaf)
         for(int i = root->len_keys; i >= 0; i--) 
            glbptr = linking((node *)root->pointers[i], glbptr);
    else{
        root->pointers[order - 1] = glbptr;
        glbptr = root;
    }
    return glbptr;
}

__global__ void linkleafs(node *root){
    node *glbptr = linking(root, NULL);
}

void operations (int n, int m, int q, node *root, int** queries, char* a, int htemph){
    char *outputfilename = a;
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for ( int i=0; i<q; i++ )  {
        int temp1 = queries[i][0], temp2 = queries[i][1];
        if(temp1 == 1){
            int *temparr, *results;
            cudaMalloc(&temparr, (temp2 + 2)*sizeof(int) );
            cudaMalloc(&results, temp2*m*sizeof(int) );
            cudaMemcpy(temparr, queries[i], (temp2 + 2)*sizeof(int), cudaMemcpyHostToDevice);
            searchkernel <<<temp2, 1>>>(root, temparr, m, results);
            int* hresults = (int *) malloc ( temp2*m* sizeof (int ) );
            cudaMemcpy(hresults, results,  temp2*m*sizeof(int), cudaMemcpyDeviceToHost);
            for ( int j=0; j< temp2; j++ ){
                if(hresults[j*m] == -1){
                    fprintf( outputfilepointer, "%d\n", hresults[j*m]);
                }
                else{
                    for(int k = 0; k < m; k++){
                        fprintf( outputfilepointer, "%d ", hresults[j*m+k]);
                    }
                    fprintf( outputfilepointer, "\n");
                }
            }
        }
        else if(temp1 == 2){
            int *temparr, *found, *hfound, *results, *hresults, count = 0;
            hfound = (int *) malloc ( (temp2+1)* sizeof (int ) );
            cudaMalloc(&found, (temp2+1)*sizeof(int));
            cudaMalloc(&temparr, (2*temp2 + 2)*sizeof(int) );
            cudaMemcpy(temparr, queries[i], (2*temp2 + 2)*sizeof(int), cudaMemcpyHostToDevice);
            rangekernelcount<<<temp2, 1>>>(root, temparr, m, found);
            cudaMemcpy(hfound, found,  (temp2+1)*sizeof(int), cudaMemcpyDeviceToHost);
            for(int j = 0; j < temp2+1; j++){
                count += hfound[j];
                if (j != 0) hfound[j] += hfound[j-1];
                else hfound[j] = 0;
            }
            cudaMemcpy(found, hfound, (temp2+1)*sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc(&results, (count*m)*sizeof(int));
            hresults = (int *) malloc ( (count*m)* sizeof (int ) );
            rangekernel<<<temp2, 1>>>(root, temparr, n, m, found, results);
            cudaMemcpy(hresults, results,  (count*m)*sizeof(int), cudaMemcpyDeviceToHost);
            for ( int j=0; j < temp2; j++ ){
                for(int k = hfound[j]; k < hfound[j+1]; k++){
                    if(hresults[k*m] == 0){}
                    else if(hresults[k*m] == -1){
                        fprintf( outputfilepointer, "%d\n", hresults[k*m]);
                        continue;
                    }
                    else{
                        for(int x = k*m; x < k*m+m; x++) fprintf( outputfilepointer, "%d ", hresults[x]);
                        fprintf( outputfilepointer, "\n");
                    }   
                }
            }
        }
        else if(temp1 == 3){
            int *temparr;
            cudaMalloc(&temparr, (3*temp2 + 2)*sizeof(int) );
            cudaMemcpy(temparr, queries[i], (3*temp2 + 2)*sizeof(int), cudaMemcpyHostToDevice);
            additionkernel<<<temp2, 1>>>(root, temparr, m);
        }
        else{
            int theight = htemph, *results, *hresults;
            cudaMalloc(&results, ((theight+1)*8+1)*sizeof(int));
            hresults = (int *) malloc (((theight+1)*8+1)*sizeof(int));
            pathkernel<<<1, 1>>>(root, temp2, results);
            cudaMemcpy(hresults, results,  ((theight+1)*8+1)*sizeof(int), cudaMemcpyDeviceToHost);
            for(int j = 0; j < (theight+1)*8+1; j++){
                if(hresults[j] == -1) break;
                fprintf( outputfilepointer, "%d ", hresults[j]);
            }
            fprintf( outputfilepointer, "\n");
        }
    }

    fclose( outputfilepointer );
}


int main(int argc,char **argv){
    node *root;
    // char instruction;

    root = NULL;

    //variable declarations
    int n, m, q, temp1, temp2;
    
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
    
    
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of rows
    fscanf( inputfilepointer, "%d", &m );      //scaning for number of columns

    // scanning for speeds of each vehicles for every subsequent toll tax combinations
    for(int i = 0; i < n; i++){
        int *hvalue = (int *) malloc ((m)* sizeof(int));
        for ( int j=0; j < m; j++ )  {
            fscanf( inputfilepointer, "%d", &hvalue[j] );
        }
        root = insertbptree(root, hvalue[0], hvalue);
    }
    //copying to GPU entire B+ tree
    node *gpuroot = copybptree(root, m);
    cudaDeviceSynchronize();
    linkleafs<<<1, 1>>>(gpuroot);
    cudaDeviceSynchronize();


    int *temph, *htemph;
    htemph = (int *)malloc(2*sizeof(int));
    cudaMalloc(&temph, (2)*sizeof(int));
    heightkernel <<<1, 1>>> (gpuroot, temph);
    cudaMemcpy(htemph, temph, 2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    fscanf( inputfilepointer, "%d", &q );      //scaning for number of queries
    int **queries = (int **) malloc ( (q)* sizeof (int *) );
    for ( int i=0; i< q; i++ )  {
        fscanf( inputfilepointer, "%d", &temp1 );
        if(temp1 == 1){
            fscanf( inputfilepointer, "%d", &temp2 );
            queries[i] = (int *) malloc ( (temp2 + 2) * sizeof (int) );
            queries[i][0] = temp1;
            queries[i][1] = temp2;
            for(int j = 2; j < temp2+2; j++){
                fscanf( inputfilepointer, "%d", &queries[i][j]);
            }
        }
        else if(temp1 == 2){
            fscanf( inputfilepointer, "%d", &temp2 );
            queries[i] = (int *) malloc ( (2*temp2 + 2) * sizeof (int) );
            queries[i][0] = temp1;
            queries[i][1] = temp2;
            for(int j = 2; j < 2*temp2+2; j++){
                fscanf( inputfilepointer, "%d", &queries[i][j]);
            }
        }
        else if(temp1 == 3){
            fscanf( inputfilepointer, "%d", &temp2 );
            queries[i] = (int *) malloc ( (3*temp2 + 2) * sizeof (int) );
            queries[i][0] = temp1;
            queries[i][1] = temp2;
            for(int j = 2; j < 3*temp2+2; j++){
                fscanf( inputfilepointer, "%d", &queries[i][j]);
            }
        }
        else{
            queries[i] = (int *) malloc ( (2) * sizeof (int) );
            queries[i][0] = temp1;
            fscanf( inputfilepointer, "%d", &queries[i][1]);
        }
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);
    operations ( n, m, q, gpuroot, queries, argv[2], htemph[0]);

    cudaDeviceSynchronize();

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    fclose( inputfilepointer );
    return 0;
}