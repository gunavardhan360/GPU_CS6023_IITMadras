#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

float* cpu_score_docs_queries_unigrams(float *scores_vectors, float *all_doc_vectors, float *all_query_vectors, int count_vocab, int count_queries, int count_docs){
    for(int i = 0; i < count_queries; i++){
        for(int j = 0; j < count_docs; j++){
            float norm_query = 0.0;
            float norm_doc = 0.0;
            float dot_prod = 0.0;
            int q = i*count_vocab;
            int d = j*count_vocab;
            for(int k = 0; k < count_vocab; k++){
                norm_query += all_query_vectors[q + k]*all_query_vectors[q + k];
		        norm_doc += all_doc_vectors[d + k]*all_doc_vectors[d + k];
		        dot_prod += all_doc_vectors[d + k]*all_query_vectors[q + k];
            }
            scores_vectors[i*count_docs + j] =  (dot_prod*dot_prod)/(norm_query*norm_doc);
        }
    }
    return scores_vectors;
}

float *cpu_score_docs_queries(float *result, float *a, float *b, int count_vocab){
    float norm_query = 0.0;
    float norm_doc = 0.0;
    float dot_prod = 0.0;
    for(int k = 0; k < count_vocab; k++){
        norm_doc += b[k]*b[k];
        norm_query += a[k]*a[k];
        dot_prod += a[k]*b[k];
    }
    result[0] = sqrt(norm_doc);
    result[1] = sqrt(norm_query);
    result[2] = dot_prod;
    return result;
}

float *cpu_query_precision(float *result, int *query_doc_IDs_ordered, int *true_doc_IDs, int len_true, int k){
    float related = 0.0;
    for(int i = 0; i < k; i++){
        for(int j = 0; j < len_true; j++){
            if(query_doc_IDs_ordered[i] == true_doc_IDs[j]){
                related++;
                break;
            }
        }
    }
    result[0] = related/k;
    return result;
}

float *all_cpu_query_precision(float *precision_arr, int *query_doc_IDs_ordered, int *true_doc_IDs, int *len_true_doc_IDs, int k, int count_docs, int len_pre){
    float related;
    int start, end, temp, s;
    for(int x = 0; x < len_pre; x++){
        related = 0.0;
        start = len_true_doc_IDs[x];
        end = len_true_doc_IDs[x+1];
        temp = x*count_docs;
        for(int i = 0; i < k; i++){
            s = query_doc_IDs_ordered[temp + i]; 
            for(int j = start; j < end; j++){
                if(s == true_doc_IDs[j]){
                    related++;
                    break;
                }
            }
        }
        precision_arr[x] = related/k;
    }
    return precision_arr;
}

float *cpu_query_recall(float *result, int *query_doc_IDs_ordered, int *true_doc_IDs, int len_true, int k){
    float related = 0.0;
    for(int i = 0; i < k; i++){
        for(int j = 0; j < len_true; j++){
            if(query_doc_IDs_ordered[i] == true_doc_IDs[j]){
                related++;
                break;
            }
        }
    }
    result[0] = related/len_true;
    return result;
}

float *all_cpu_query_recall(float *recall_arr, int *query_doc_IDs_ordered, int *true_doc_IDs, int *len_true_doc_IDs, int k, int count_docs, int len_pre){
    float related;
    int start, end, temp, s;
    for(int x = 0; x < len_pre; x++){
        related = 0.0;
        start = len_true_doc_IDs[x];
        end = len_true_doc_IDs[x+1];
        temp = x*count_docs;
        for(int i = 0; i < k; i++){
            s = query_doc_IDs_ordered[temp + i]; 
            for(int j = start; j < end; j++){
                if(s == true_doc_IDs[j]){
                    related++;
                    break;
                }
            }
        }
        recall_arr[x] = related/(end-start);
    }
    return recall_arr;
}

float *cpu_query_fscore(float *result, int *query_doc_IDs_ordered, int *true_doc_IDs, int len_true, int k){
    float related = 0.0;
    for(int i = 0; i < k; i++){
        for(int j = 0; j < len_true; j++){
            if(query_doc_IDs_ordered[i] == true_doc_IDs[j]){
                related++;
                break;
            }
        }
    }
    float r = related/len_true;
    float p = related/k;
    if (p*r == 0)
        result[0] = 0;
    else
        result[0] = (2*r*p)/(r+p);
    return result;
}

float *all_cpu_query_fscore(float *fscore_arr, int *query_doc_IDs_ordered, int *true_doc_IDs, int *len_true_doc_IDs, int k, int count_docs, int len_pre){
    float related, r, p;
    int start, end, temp, s;
    for(int x = 0; x < len_pre; x++){
        related = 0.0;
        start = len_true_doc_IDs[x];
        end = len_true_doc_IDs[x+1];
        temp = x*count_docs;
        for(int i = 0; i < k; i++){
            s = query_doc_IDs_ordered[temp + i]; 
            for(int j = start; j < end; j++){
                if(s == true_doc_IDs[j]){
                    related++;
                    break;
                }
            }
        }
        r = related/(end-start);
        p = related/k;
        if (p*r == 0)
            fscore_arr[x] = 0;
        else
            fscore_arr[x] = (2*r*p)/(r+p);
    }
    return fscore_arr;
}

float *cpu_query_MAP(float *result, int *query_doc_IDs_ordered, int *true_doc_IDs, int len_true, int k){
    float related = 0.0;
    float sum = 0;
    for(int i = 0; i < k; i++){
        for(int j = 0; j < len_true; j++){
            if(query_doc_IDs_ordered[i] == true_doc_IDs[j]){
                related++;
				sum += related/(i+1);
                break;
            }
        }
    }
	if(related == 0)
		result[0] = 0;
	else
		result[0] = sum/related;
    return result;
}

float *all_cpu_query_MAP(float *MAP_arr, int *query_doc_IDs_ordered, int *true_doc_IDs, int *len_true_doc_IDs, int k, int count_docs, int len_pre){
    float related, r, p;
    int start, end, temp, s, sum;
    for(int x = 0; x < len_pre; x++){
        related = 0.0;
        start = len_true_doc_IDs[x];
        end = len_true_doc_IDs[x+1];
        sum = 0;
        temp = x*count_docs;
        for(int i = 0; i < k; i++){
            s = query_doc_IDs_ordered[temp + i]; 
            for(int j = start; j < end; j++){
                if(s == true_doc_IDs[j]){
                    related++;
                    sum += related/(i+1);
                    break;
                }
            }
        }
        if(related == 0)
            MAP_arr[x] = 0;
        else
            MAP_arr[x] = sum/related;
    }
    return MAP_arr;
}

float* cpu_score_docs_queries_bigrams(float *scores_vectors, float *all_doc_vectors, float *all_query_vectors, int count_vocab, int count_queries, int count_docs){
    for(int i = 0; i < count_queries; i++){
        for(int j = 0; j < count_docs; j++){
            float norm_query = 0.0;
            float norm_doc = 0.0;
            float dot_prod = 0.0;
            int q = i*count_vocab;
            int d = j*count_vocab;
            for(int k = 0; k < count_vocab; k++){
                norm_query += all_query_vectors[q + k]*all_query_vectors[q + k];
		        norm_doc += all_doc_vectors[d + k]*all_doc_vectors[d + k];
		        dot_prod += all_doc_vectors[d + k]*all_query_vectors[q + k];
            }
            scores_vectors[i*count_docs + j] +=  (dot_prod*dot_prod)/(norm_query*norm_doc);
        }
    }
    return scores_vectors;
}

int main(){
    return 0;
}