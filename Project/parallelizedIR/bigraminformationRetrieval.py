from util import *

# Add your import statements here
import pickle
import time
import datetime as dt

mod = SourceModule("""
__global__ void score_docs_queries(float *scores_vectors, float *doc_vectors, float *query_vectors, int count_docs, int count_vocab)
{
  const int i = threadIdx.x;
  const int j = blockIdx.x;
  float norm_query = 0.0;
  float norm_doc = 0.0;
  float dot_prod = 0.0;
  int q = i*count_vocab;
  int d = j*count_vocab;
  __shared__ float x, y;
  for(int k = 0; k < count_vocab; k++){
	  x = query_vectors[q + k];
	  y = doc_vectors[d + k];
	  norm_query += x*x;
	  norm_doc += y*y;
	  dot_prod += x*y;
  }
  scores_vectors[i*count_docs + j] =  (dot_prod*dot_prod)/(norm_query*norm_doc);
}

__global__ void score_docs_queries_block(float *scores_vectors, float *doc_vectors, float *query_vectors, int count_docs, int count_vocab, int num_queries)
{
  const int i = (blockIdx.x) % num_queries;
  const int j = blockIdx.x / num_queries;
  float norm_query = 0.0;
  float norm_doc = 0.0;
  float dot_prod = 0.0;
  int q = i*count_vocab;
  int d = j*count_vocab;
  __shared__ float x, y;
  for(int k = 0; k < count_vocab; k++){
	  x = query_vectors[q + k];
	  y = doc_vectors[d + k];
	  norm_query += x*x;
	  norm_doc += y*y;
	  dot_prod += x*y;
  }
  scores_vectors[i*count_docs + j] =  (dot_prod*dot_prod)/(norm_query*norm_doc);
}

__global__ void score_docs_queries_singlethread(float *scores_vectors, float *doc_vectors, float *query_vectors, int count_queries, int count_docs, int count_vocab)
{
  __shared__ float x, y;
  float norm_query, norm_doc, dot_prod;
  int q, d;
  for(int i = 0; i < count_queries; i++){
	  	for(int j = 0; j < count_docs; j++){
			norm_query = 0.0;
			norm_doc = 0.0;
			dot_prod = 0.0;
			q = i*count_vocab;
			d = j*count_vocab;
			for(int k = 0; k < count_vocab; k++){
				x = query_vectors[q + k];
				y = doc_vectors[d + k];
				norm_query += x*x;
				norm_doc += y*y;
				dot_prod += x*y;
			}
			scores_vectors[i*count_docs + j] =  (dot_prod*dot_prod)/(norm_query*norm_doc);
		}
  	}
}

__global__ void high_score_docs_queries(float *scores_vectors, float *temp, float *doc_vectors, float *query_vectors, int count_docs, int count_vocab)
{
  const int i = threadIdx.x;
  const int j = blockIdx.x;
  float norm_query = 0.0;
  float norm_doc = 0.0;
  float dot_prod = 0.0;
  int q = i*count_vocab;
  int d = j*count_vocab;
  __shared__ float x, y;
  for(int k = 0; k < count_vocab; k++){
	  x = query_vectors[q + k];
	  y = doc_vectors[d + k];
	  norm_query += x*x;
	  norm_doc += y*y;
	  dot_prod += x*y;
  }
  scores_vectors[i*count_docs + j] = temp[i*count_docs + j] + (dot_prod*dot_prod)/(norm_query*norm_doc);
}

__global__ void high_score_docs_queries_block(float *scores_vectors, float *temp, float *doc_vectors, float *query_vectors, int count_docs, int count_vocab, int num_queries)
{
  const int i = (blockIdx.x) % num_queries;
  const int j = blockIdx.x / num_queries;
  float norm_query = 0.0;
  float norm_doc = 0.0;
  float dot_prod = 0.0;
  int q = i*count_vocab;
  int d = j*count_vocab;
  __shared__ float x, y;
  for(int k = 0; k < count_vocab; k++){
	  x = query_vectors[q + k];
	  y = doc_vectors[d + k];
	  norm_query += x*x;
	  norm_doc += y*y;
	  dot_prod += x*y;
  }
  scores_vectors[i*count_docs + j] = temp[i*count_docs + j] + (dot_prod*dot_prod)/(norm_query*norm_doc);
}
""")

score_docs_queries = mod.get_function("score_docs_queries")
score_docs_queries_block = mod.get_function("score_docs_queries_block")
high_score_docs_queries = mod.get_function("high_score_docs_queries")
high_score_docs_queries_block = mod.get_function("high_score_docs_queries_block")
score_docs_queries_singlethread = mod.get_function("score_docs_queries_singlethread")

class InformationRetrievalbigram():

	def __init__(self, args):
		self.index = None
		self.args = args

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		
		index = None
		self.index = index
		
		D = len(docs)
        
		dict_list_of_words = {}
		vocab = []
		for document in docs:
			for sentence in document:
				for word in sentence:
					vocab.append(word)

		for document in docs:
			for sentence in document:
				for i in range(len(sentence)-1):
					word = sentence[i] + sentence[i+1]
					if word not in dict_list_of_words:
						dict_list_of_words[word] = 1
					else:
						dict_list_of_words[word] += 1	#this can have repetition of words

		for key in dict_list_of_words:
			if dict_list_of_words[key] > 1:
				vocab.append(key)
		self.list_unique = list(set(vocab))

		start = dt.datetime.now()
		df = np.zeros(len(self.list_unique)) # df(i) is number of docs containing term i, used for calculating IDF 
		TD_matrix = np.zeros([len(self.list_unique),len(docs)]) #term-document matrix
        
        
		for j, doc in enumerate(docs):       # iterate over documents
			for k, sentence in enumerate(doc):  # iterate over sentences for a document
				for word in sentence:
					try:
						TD_matrix[self.list_unique.index(word),j] += 1
					except:
						temp_skip = 0

		for j, doc in enumerate(docs):       # iterate over documents
			for k, sentence in enumerate(doc):  # iterate over sentences for a document
				for i in range(len(sentence)-1):
					word = sentence[i] + sentence[i+1]
					try:
						TD_matrix[self.list_unique.index(word),j] += 1
					except:
						temp_skip = 0
        
		df = np.sum(TD_matrix > 0, axis=1)

		self.IDF = np.log(D/df)          

		self.doc_weights = np.zeros([len(self.list_unique),len(docs)])
        
		for i in range(len(self.list_unique)):
			self.doc_weights[i,:] = self.IDF[i]*TD_matrix[i,:]   # vector weights for each document 
                      

		index = {key: None for key in docIDs}  # initialize dictionary with keys as doc_IDs
       
		for j in range(len(docs)): 
			index[docIDs[j]] = self.doc_weights[:,j]   # update dict-values with weight vector for corresponding docIDs                
        
		self.index = index

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		TQ_matrix = np.zeros([len(self.list_unique),len(queries)]) # term frequency matrix (for each query in list queries)
        
		for i, unique_word in enumerate(self.list_unique):
			for j, query in enumerate(queries):       # iterate over all queries
				for k, sentence in enumerate(query):  # iterate over sentences for a query
					for word in sentence:
						if unique_word == word:
							TQ_matrix[i,j] += 1 

		for i, unique_word in enumerate(self.list_unique):
			for j, query in enumerate(queries):       # iterate over all queries
				for k, sentence in enumerate(query):  # iterate over sentences for a query
					for z in range(len(sentence)-1):
						if unique_word == sentence[z] + sentence[z+1]:
							TQ_matrix[i,j] += 1 

		self.query_weights = np.zeros([len(self.list_unique),len(queries)])
        
		for i, unique_word in enumerate(self.list_unique):
			self.query_weights[i,:] = self.IDF[i]*TQ_matrix[i,:]  # vector weights for each query  
        
		id_docs = list(self.index.keys())
        
        
		doc_IDs_ordered  = list(range(len(queries)))

		tic = time.perf_counter()
		num_vocab_elements = len(self.doc_weights[:,0])
		
		if self.args.runtime == "GPU" and num_vocab_elements < 30000:
			num_queries = len(queries)
			all_query_vectors = []
			for j in range(num_queries):
				all_query_vectors.extend(self.query_weights[:,j])
			all_doc_vectors = []
			for doc_id, doc_vector in self.index.items():
				all_doc_vectors.extend(doc_vector)
			num_documents = len(id_docs)

			scores_vectors = [0]*num_queries*num_documents
			
			scores_vectors = np.array(scores_vectors).astype(np.float32)
			all_doc_vectors = np.array(all_doc_vectors).astype(np.float32)
			all_query_vectors = np.array(all_query_vectors).astype(np.float32)

			if num_queries < 1024:
				score_docs_queries(drv.Out(scores_vectors), drv.In(all_doc_vectors), drv.In(all_query_vectors), 
											np.intc(num_documents), np.intc(num_vocab_elements), block=(num_queries,1,1), grid=(num_documents,1))
			else:
				score_docs_queries_block(drv.Out(scores_vectors), drv.In(all_doc_vectors), drv.In(all_query_vectors), 
											np.intc(num_documents), np.intc(num_vocab_elements), np.intc(num_queries), block=(1,1,1), grid=(num_documents*num_queries,1))
		
			for i in range(num_queries):
				dict_cosine = {key: None for key in id_docs} 
				for j in range(num_documents):
					dict_cosine[j+1] = math.sqrt(scores_vectors[i*num_documents+j])
				dc_sort = sorted(dict_cosine.items(),key = operator.itemgetter(1),reverse = True)
				doc_IDs_ordered[i] = [x for x, _ in dc_sort]

		elif self.args.runtime == "GPU":
			scores_vectors = [0]*num_queries*num_documents
			scores_vectors = np.array(scores_vectors).astype(np.float32)
			step = 16000
			for i in range(0, num_vocab_elements, step):
				temp = scores_vectors	
				start = i
				if start + step > num_vocab_elements:
					end = num_vocab_elements
				else:
					end = start + step
				all_query_vectors = []
				for j in range(num_queries):
					all_query_vectors.extend(self.query_weights[:,j][start:end])
				all_doc_vectors = []
				for doc_id, doc_vector in self.index.items():
					all_doc_vectors.extend(doc_vector[start:end])

				all_doc_vectors = np.array(all_doc_vectors).astype(np.float32)
				all_query_vectors = np.array(all_query_vectors).astype(np.float32)

				if num_queries < 1024:
					high_score_docs_queries(drv.Out(scores_vectors), drv.In(temp), drv.In(all_doc_vectors), drv.In(all_query_vectors), 
												np.intc(num_documents), np.intc(num_vocab_elements), block=(num_queries,1,1), grid=(num_documents,1))
				else:
					high_score_docs_queries_block(drv.Out(scores_vectors), drv.In(temp), drv.In(all_doc_vectors), drv.In(all_query_vectors), 
												np.intc(num_documents), np.intc(num_vocab_elements), np.intc(num_queries), block=(1,1,1), grid=(num_documents*num_queries,1))

			for i in range(num_queries):
				dict_cosine = {key: None for key in id_docs} 
				for j in range(num_documents):
					dict_cosine[j+1] = math.sqrt(scores_vectors[i*num_documents+j])
				dc_sort = sorted(dict_cosine.items(),key = operator.itemgetter(1),reverse = True)
				doc_IDs_ordered[i] = [x for x, _ in dc_sort]

		elif self.args.runtime == "CPU":
			num_vocab_elements = len(self.doc_weights[:,0])

			for j in range(len(queries)):
				dict_cosine = {key: None for key in id_docs} # given ONE query, stores cosine measures for between query and all docs
				for doc_id, doc_vector in self.index.items():
					a = doc_vector
					b = self.query_weights[:,j]
					vector_doc = np.array(a).astype(np.float32)
					vector_query = np.array(b).astype(np.float32)
					temp = np.array([0]*3).astype(np.float32)
					ccode.cpu_score_docs_queries.restype = ndpointer(dtype=c_float,
                          			shape=(3,))
					temp = ccode.cpu_score_docs_queries(c_void_p(temp.ctypes.data), c_void_p(vector_query.ctypes.data), c_void_p(vector_doc.ctypes.data), c_int(num_vocab_elements))
					dict_cosine[doc_id] = temp[2]/(temp[0]*temp[1])

				dc_sort = sorted(dict_cosine.items(),key = operator.itemgetter(1),reverse = True)
				doc_IDs_ordered[j] = [x for x, _ in dc_sort]
			
		elif self.args.runtime == "python":
			for j in range(len(queries)):
				dict_cosine = {key: None for key in id_docs} # given ONE query, stores cosine measures for between query and all docs
				for doc_id, doc_vector in self.index.items():
					a = doc_vector
					b = self.query_weights[:,j]
					dict_cosine[doc_id] = dot(a,b)/(norm(a)*norm(b))
				dc_sort = sorted(dict_cosine.items(),key = operator.itemgetter(1),reverse = True)
				doc_IDs_ordered[j] = [x for x, _ in dc_sort]
		
		elif self.args.runtime == "spython":
			for j in range(len(queries)):
				dict_cosine = {key: None for key in id_docs} # given ONE query, stores cosine measures for between query and all docs
				for doc_id, doc_vector in self.index.items():
					a = doc_vector
					b = self.query_weights[:,j]
					n_q = 0.0
					n_d = 0.0
					d_p = 0.0
					for i in range(num_vocab_elements):
						n_d += a[i]*a[i]
						n_q += b[i]*b[i]
						d_p += a[i]*b[i]
					dict_cosine[doc_id] = d_p/((n_q*n_d)**0.5)
				dc_sort = sorted(dict_cosine.items(),key = operator.itemgetter(1),reverse = True)
				doc_IDs_ordered[j] = [x for x, _ in dc_sort]

		
		
		toc = time.perf_counter()
		print(f"IR search operation in {toc - tic:0.4f} seconds")
		return doc_IDs_ordered
