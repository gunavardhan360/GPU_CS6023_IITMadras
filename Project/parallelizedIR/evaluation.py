from util import *

# CUDA code
mod = SourceModule("""
__global__ void query_Precision(float *dest, int *query_doc_IDs_ordered, int *true_doc_IDs, int len_true_doc_IDs, int k)
{
	
	float retreived = 0.0;
	int related = 0;
	__shared__ int s;
	for(int i = 0; i < k; i++){
		retreived++;
		s = query_doc_IDs_ordered[i];
		for(int j = 0; j < len_true_doc_IDs; j++){
			if(s == true_doc_IDs[j]){
				related++;
				break;
			}
		}
	}
	dest[0] = related/retreived;
}

__global__ void all_query_Precision(float *precision_arr, int *query_doc_IDs_ordered, int *true_doc_IDs, int *len_true_doc_IDs, int k, int count_docs)
{
	const int x = blockIdx.x;
	float related = 0.0;
	int start = len_true_doc_IDs[x];
	int end = len_true_doc_IDs[x+1];
	int temp = x*count_docs;
	__shared__ int s;
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

__global__ void query_Recall(float *dest, int *query_doc_IDs_ordered, int *true_doc_IDs, int len_true_doc_IDs, int k)
{
	
	float retreived = 0.0;
	int related = 0;
	__shared__ int s;
	for(int i = 0; i < k; i++){
		retreived++;
		s = query_doc_IDs_ordered[i];
		for(int j = 0; j < len_true_doc_IDs; j++){
			if(s == true_doc_IDs[j]){
				related++;
				break;
			}
		}
	}
	dest[0] = related/retreived;
}

__global__ void all_query_Recall(float *recall_arr, int *query_doc_IDs_ordered, int *true_doc_IDs, int *len_true_doc_IDs, int k, int count_docs)
{
	const int x = blockIdx.x;
	float related = 0.0;
	int start = len_true_doc_IDs[x];
	int end = len_true_doc_IDs[x+1];
	int relevant = end - start;
	int temp = x*count_docs;
	__shared__ int s;
	for(int i = 0; i < k; i++){
		s = query_doc_IDs_ordered[temp + i]; 
		for(int j = start; j < end; j++){
			if(s == true_doc_IDs[j]){
				related++;
				break;
			}
		}
	}
	recall_arr[x] = related/relevant;
}

__global__ void all_query_fscore(float *fscore_arr, int *query_doc_IDs_ordered, int *true_doc_IDs, int *len_true_doc_IDs, int k, int count_docs)
{
	const int x = blockIdx.x;
	float related = 0.0;
	int start = len_true_doc_IDs[x];
	int end = len_true_doc_IDs[x+1];
	int relevant = end - start;
	int temp = x*count_docs;
	__shared__ int s;
	for(int i = 0; i < k; i++){
		s = query_doc_IDs_ordered[temp + i]; 
		for(int j = start; j < end; j++){
			if(s == true_doc_IDs[j]){
				related++;
				break;
			}
		}
	}
	float p = related/relevant;
	float r = related/k;

	if(p*r == 0)
		fscore_arr[x] = 0;
	else
		fscore_arr[x] = 2*p*r/(p+r);
}

__global__ void all_query_MAP(float *MAP_arr, int *query_doc_IDs_ordered, int *true_doc_IDs, int *len_true_doc_IDs, int k, int count_docs)
{
	const int x = blockIdx.x;
	float related = 0.0;
	int start = len_true_doc_IDs[x];
	int end = len_true_doc_IDs[x+1];
	float sum = 0;
	int temp = x*count_docs;
	__shared__ int s;
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
""")

query_Precision = mod.get_function("query_Precision")
query_Recall = mod.get_function("query_Recall")
all_query_Precision = mod.get_function("all_query_Precision")
all_query_Recall = mod.get_function("all_query_Recall")
all_query_fscore = mod.get_function("all_query_fscore")
all_query_MAP = mod.get_function("all_query_MAP")

class Evaluation():
	def __init__(self, args):
		self.args = args

	def queryPrecision(self, query_doc_IDs_ordered, true_doc_IDs, len_true_doc_IDs, k):

		len_pre = len(len_true_doc_IDs)-1
		precision_arr = np.array([0]*len_pre).astype(np.float32)

		for x in range(len_pre):
			related = 0.0
			start = len_true_doc_IDs[x]
			end = len_true_doc_IDs[x+1]
			for i in range(k):
				m = query_doc_IDs_ordered[x*1400 + i]
				for j in range(start, end):
					n = true_doc_IDs[j]
					if query_doc_IDs_ordered[x*1400 + i] == true_doc_IDs[j]:
						related += 1.0
						break
			precision_arr[x] = related/k

		return precision_arr


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		count = 0
		total_query_doc_ids = []
		total_true_doc_ids = []
		len_query_doc_ids = 0
		len_true_doc_ids = [0]
		for i,query_doc_IDs_ordered in enumerate(doc_IDs_ordered):
			len_query_doc_ids = len(query_doc_IDs_ordered)
			true_doc_IDs = []
            
			while int(qrels[count]["query_num"]) == query_ids[i]:
				true_doc_IDs.append(int(qrels[count]["id"]))
				count += 1
				if count == len(qrels):
					break

			total_query_doc_ids.extend(query_doc_IDs_ordered)
			total_true_doc_ids.extend(true_doc_IDs)
			len_true_doc_ids.append(len(true_doc_IDs) + len_true_doc_ids[-1])

		len_pre = len(len_true_doc_ids)-1
		precision_arr = np.array([0]*len_pre).astype(np.float32)
		total_qdocs_ordered = np.array(total_query_doc_ids).astype(np.int32)
		total_tdoc_IDs = np.array(total_true_doc_ids).astype(np.int32)
		len_tdoc_IDs = np.array(len_true_doc_ids).astype(np.int32)
		
		if self.args.runtime == "GPU":
			all_query_Precision(drv.Out(precision_arr), drv.In(total_qdocs_ordered), drv.In(total_tdoc_IDs), drv.In(len_tdoc_IDs), np.intc(k), np.intc(len_query_doc_ids), block=(1,1,1), grid=(len_pre,1))
		elif self.args.runtime == "CPU":
			ccode.cpu_query_precision.restype = ndpointer(dtype=c_float, shape=(2,))
			for x in range(len_pre):
				start = len_true_doc_ids[x]
				end = len_true_doc_ids[x+1]
				qdocs_ordered = np.array(total_query_doc_ids[x*len_query_doc_ids:x*len_query_doc_ids+len_query_doc_ids]).astype(np.int32)
				tdoc_IDs = np.array(total_true_doc_ids[start:end]).astype(np.int32)
				result = np.array([0]*2).astype(np.float32)

				result = ccode.cpu_query_precision(c_void_p(result.ctypes.data), c_void_p(qdocs_ordered.ctypes.data), c_void_p(tdoc_IDs.ctypes.data), c_int(len(tdoc_IDs)), c_int(k))
				precision_arr[x] = result[0]
		elif self.args.runtime == "python":
			precision_arr = self.queryPrecision(total_query_doc_ids, total_true_doc_ids, len_true_doc_ids, k)

		result = np.sum(precision_arr)
		meanPrecision = result/len(query_ids)
		return meanPrecision

	def queryRecall(self, query_doc_IDs_ordered, true_doc_IDs, len_true_doc_IDs, k):

		len_pre = len(len_true_doc_IDs)-1
		recall_arr = np.array([0]*len_pre).astype(np.float32)

		for x in range(len_pre):
			related = 0.0
			start = len_true_doc_IDs[x]
			end = len_true_doc_IDs[x+1]
			relevent = end - start
			for i in range(k):
				m = query_doc_IDs_ordered[x*1400 + i]
				for j in range(start, end):
					n = true_doc_IDs[j]
					if query_doc_IDs_ordered[x*1400 + i] == true_doc_IDs[j]:
						related += 1.0
						break
			recall_arr[x] = related/relevent

		return recall_arr


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""


		meanRecall = -1
		#Fill in code here
		total_query_doc_ids = []
		total_true_doc_ids = []
		len_query_doc_ids = 0
		len_true_doc_ids = [0]
		count = 0
		for i,query_doc_IDs_ordered in enumerate(doc_IDs_ordered):
			len_query_doc_ids = len(query_doc_IDs_ordered)
			true_doc_IDs = []
			while int(qrels[count]["query_num"]) == query_ids[i]:
				true_doc_IDs.append(int(qrels[count]["id"]))
				count += 1
				
				if count == len(qrels):
					break

			total_query_doc_ids.extend(query_doc_IDs_ordered)
			total_true_doc_ids.extend(true_doc_IDs)
			len_true_doc_ids.append(len(true_doc_IDs) + len_true_doc_ids[-1])
		
		len_pre = len(len_true_doc_ids)-1
		recall_arr = np.array([0]*len_pre).astype(np.float32)
		total_qdocs_ordered = np.array(total_query_doc_ids).astype(np.int32)
		total_tdoc_IDs = np.array(total_true_doc_ids).astype(np.int32)
		len_tdoc_IDs = np.array(len_true_doc_ids).astype(np.int32)

		if self.args.runtime == "GPU":
			all_query_Recall(drv.Out(recall_arr), drv.In(total_qdocs_ordered), drv.In(total_tdoc_IDs), drv.In(len_tdoc_IDs), np.intc(k), np.intc(len_query_doc_ids), block=(1,1,1), grid=(len_pre,1))
		elif self.args.runtime == "CPU":
			ccode.cpu_query_recall.restype = ndpointer(dtype=c_float, shape=(2,))
			for x in range(len_pre):
				start = len_true_doc_ids[x]
				end = len_true_doc_ids[x+1]
				qdocs_ordered = np.array(total_query_doc_ids[x*len_query_doc_ids:x*len_query_doc_ids+len_query_doc_ids]).astype(np.int32)
				tdoc_IDs = np.array(total_true_doc_ids[start:end]).astype(np.int32)
				result = np.array([0]*2).astype(np.float32)

				result = ccode.cpu_query_recall(c_void_p(result.ctypes.data), c_void_p(qdocs_ordered.ctypes.data), c_void_p(tdoc_IDs.ctypes.data), c_int(len(tdoc_IDs)), c_int(k))
				recall_arr[x] = result[0]
		elif self.args.runtime == "python":
			recall_arr = self.queryRecall(total_query_doc_ids, total_true_doc_ids, len_true_doc_ids, k)

		result = np.sum(recall_arr)

		meanRecall = result/len(query_ids)
		
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, true_doc_IDs, len_true_doc_IDs, k):

		len_pre = len(len_true_doc_IDs)-1
		fscore_arr = np.array([0]*len_pre).astype(np.float32)

		for x in range(len_pre):
			related = 0.0
			start = len_true_doc_IDs[x]
			end = len_true_doc_IDs[x+1]
			relevent = end - start
			for i in range(k):
				m = query_doc_IDs_ordered[x*1400 + i]
				for j in range(start, end):
					n = true_doc_IDs[j]
					if query_doc_IDs_ordered[x*1400 + i] == true_doc_IDs[j]:
						related += 1.0
						break
			r = related/relevent
			p = related/k
			if p*r == 0:
				fscore_arr[x] = 0
			else:
				fscore_arr[x] = 2*p*r/(p+r)
		return fscore_arr

	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		count = 0
		total_query_doc_ids = []
		total_true_doc_ids = []
		len_query_doc_ids = 0
		len_true_doc_ids = [0]
		for i,query_doc_IDs_ordered in enumerate(doc_IDs_ordered):
			len_query_doc_ids = len(query_doc_IDs_ordered)
			true_doc_IDs = []
			while int(qrels[count]["query_num"]) == query_ids[i]:
				true_doc_IDs.append(int(qrels[count]["id"]))
				count += 1
				if count == len(qrels):
					break

			total_query_doc_ids.extend(query_doc_IDs_ordered)
			total_true_doc_ids.extend(true_doc_IDs)
			len_true_doc_ids.append(len(true_doc_IDs) + len_true_doc_ids[-1])

		len_pre = len(len_true_doc_ids)-1
		fscore_arr = np.array([0]*len_pre).astype(np.float32)
		total_qdocs_ordered = np.array(total_query_doc_ids).astype(np.int32)
		total_tdoc_IDs = np.array(total_true_doc_ids).astype(np.int32)
		len_tdoc_IDs = np.array(len_true_doc_ids).astype(np.int32)

		if self.args.runtime == "GPU":
			all_query_fscore(drv.Out(fscore_arr), drv.In(total_qdocs_ordered), drv.In(total_tdoc_IDs), drv.In(len_tdoc_IDs), np.intc(k), np.intc(len_query_doc_ids), block=(1,1,1), grid=(len_pre,1))
		elif self.args.runtime == "CPU":
			ccode.cpu_query_fscore.restype = ndpointer(dtype=c_float, shape=(2,))
			for x in range(len_pre):
				start = len_true_doc_ids[x]
				end = len_true_doc_ids[x+1]
				qdocs_ordered = np.array(total_query_doc_ids[x*len_query_doc_ids:x*len_query_doc_ids+len_query_doc_ids]).astype(np.int32)
				tdoc_IDs = np.array(total_true_doc_ids[start:end]).astype(np.int32)
				result = np.array([0]*2).astype(np.float32)

				result = ccode.cpu_query_fscore(c_void_p(result.ctypes.data), c_void_p(qdocs_ordered.ctypes.data), c_void_p(tdoc_IDs.ctypes.data), c_int(len(tdoc_IDs)), c_int(k))
				fscore_arr[x] = result[0]
		elif self.args.runtime == "python":
			fscore_arr = self.queryFscore(total_query_doc_ids, total_true_doc_ids, len_true_doc_ids, k)

		result = np.sum(fscore_arr)
		meanFscore = result/len(query_ids)
		return meanFscore



	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		relevance = list(range(k))
		true_doc_IDs_proxy = [x for x,_ in true_doc_IDs]
		true_relevance_proxy = [y for _,y in true_doc_IDs]

		for i,docID in enumerate(query_doc_IDs_ordered):
			try:
				index = true_doc_IDs_proxy.index(docID)
				relevance[i] = true_relevance_proxy[index]
			except:
				relevance[i] = 0
			if i == k-1:
				break
				
		DCG  = 0
		for i,rel in enumerate(relevance):
			DCG += rel/math.log((i+2),2)  # formula: summation over relavance/i+1   
		
		IDCG = 0  # Ideal DCG which is calculated with relevance array sorted 
		relevance_sort = list(sorted(relevance, reverse=True))
		
		for i,rel in enumerate(relevance_sort):
			if rel == 0:
				break
			IDCG += rel/math.log((i+2),2) 
		if DCG == 0:
			nDCG = 0
		else: 
			nDCG = DCG/IDCG
		return nDCG



	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		NDCG_sum  = 0
		count = 0
		for i,query_doc_IDs_ordered in enumerate(doc_IDs_ordered):
			
			true_doc_IDs = []
			
			while int(qrels[count]["query_num"]) == query_ids[i]:
				true_doc_IDs.append((int(qrels[count]["id"]), 5-int(qrels[count]["position"])))
				count += 1
				if count == len(qrels):
					break
				
			nDCG = self.queryNDCG(query_doc_IDs_ordered, query_ids[i], true_doc_IDs, k)
			NDCG_sum += nDCG
		
		meanNDCG = NDCG_sum/len(query_ids)
		return meanNDCG

	def queryAveragePrecision(self, query_doc_IDs_ordered, true_doc_IDs, len_true_doc_IDs, k):

		len_pre = len(len_true_doc_IDs)-1
		MAP_arr = np.array([0]*len_pre).astype(np.float32)
		for x in range(len_pre):
			sum = 0
			related = 0.0
			start = len_true_doc_IDs[x]
			end = len_true_doc_IDs[x+1]
			for i in range(k):
				m = query_doc_IDs_ordered[x*1400 + i]
				for j in range(start, end):
					n = true_doc_IDs[j]
					if query_doc_IDs_ordered[x*1400 + i] == true_doc_IDs[j]:
						related += 1.0
						sum += (related/(i+1))
						break
			if(related == 0):
				MAP_arr[x] = 0
			else:
				MAP_arr[x] = sum/related

		return MAP_arr


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		count = 0
		total_query_doc_ids = []
		total_true_doc_ids = []
		len_query_doc_ids = 0
		len_true_doc_ids = [0]
		for i,query_doc_IDs_ordered in enumerate(doc_IDs_ordered):
			len_query_doc_ids = len(query_doc_IDs_ordered)
			
			true_doc_IDs = []
			
			while int(q_rels[count]["query_num"]) == query_ids[i]:
				true_doc_IDs.append(int(q_rels[count]["id"]))
				count += 1
				if count == len(q_rels):
					break
			
			total_query_doc_ids.extend(query_doc_IDs_ordered)
			total_true_doc_ids.extend(true_doc_IDs)
			len_true_doc_ids.append(len(true_doc_IDs) + len_true_doc_ids[-1])
		
		len_pre = len(len_true_doc_ids)-1
		MAP_arr = np.array([0]*len_pre).astype(np.float32)
		total_qdocs_ordered = np.array(total_query_doc_ids).astype(np.int32)
		total_tdoc_IDs = np.array(total_true_doc_ids).astype(np.int32)
		len_tdoc_IDs = np.array(len_true_doc_ids).astype(np.int32)

		if self.args.runtime == "GPU":
			all_query_MAP(drv.Out(MAP_arr), drv.In(total_qdocs_ordered), drv.In(total_tdoc_IDs), drv.In(len_tdoc_IDs), np.intc(k), np.intc(len_query_doc_ids), block=(1,1,1), grid=(len_pre,1))
		elif self.args.runtime == "CPU":
			ccode.cpu_query_MAP.restype = ndpointer(dtype=c_float, shape=(2,))
			for x in range(len_pre):
				start = len_true_doc_ids[x]
				end = len_true_doc_ids[x+1]
				qdocs_ordered = np.array(total_query_doc_ids[x*len_query_doc_ids:x*len_query_doc_ids+len_query_doc_ids]).astype(np.int32)
				tdoc_IDs = np.array(total_true_doc_ids[start:end]).astype(np.int32)
				result = np.array([0]*2).astype(np.float32)

				result = ccode.cpu_query_MAP(c_void_p(result.ctypes.data), c_void_p(qdocs_ordered.ctypes.data), c_void_p(tdoc_IDs.ctypes.data), c_int(len(tdoc_IDs)), c_int(k))
				MAP_arr[x] = result[0]
		elif self.args.runtime == "python":
			MAP_arr = self.queryAveragePrecision(total_query_doc_ids, total_true_doc_ids, len_true_doc_ids, k)
		
		result = np.sum(MAP_arr)
		meanAveragePrecision = result/len(query_ids)
		return meanAveragePrecision