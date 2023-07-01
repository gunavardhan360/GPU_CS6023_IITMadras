Code is developed by Gunavardhan Reddy CH18B035 following are the instructions to execute

To run the code 
1.Required modules : {pycuda, matplotlib, nltk, numpy}
2.Change the path in util.py on line 25 to sequential.so file present in parallelizedIR folder

Note that this code is designed to work for both Python 2 and Python 3. But recommended to execute on Python 3.9 (error might come from import statements for Python 2)

The following are file descriptions:
main.py - The main module that contains the outline of the Search Engine linking all the substeps.
preprocessing.py - The preprocessing module contains all preprocessing steps such as sentenceSegmentation, tokenization, inflectionReduction, stopwordRemoval

The following are files from parallelizedIR folder
informationRetrieval.py - The informationRetrieval module contains functions to build the index and rank the documents based on the queries which is parallelizedIR
evaluation.py - The evaluation module contains different measures of evaluating the dataset precision, recall, F-score, nDCG, meanAveragePrecision implemented in parallel
sequential.c - This file contains all the c functions of the above mentioned parallel process to compare with the cuda functions and is compiled in optimised manner

To run the code, run main.py with the appropriate arguments.
Usage: main.py [-runtime (GPU|CPU|python)] [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)] 

When the -runtime flag is passed, with appropriate argument the system runs on the designated code. For example:
> python main.py -runtime GPU
Runs using parallelization implemented on cuda
> python main.py -runtime CPU
Runs using sequentially using the functions implemented on c
> python main.py -runtime python
Runs using sequentially using the functions implemented on python using numpy library

When no flag is not passed, all the queries in the Cranfield dataset are considered and precision@k, recall@k, f-score@k, nDCG@k and the Mean Average Precision are computed.

In both the cases, *queries.txt files and *docs.txt files and graph ploting all evaluation metrics will be generated in the OUTPUT FOLDER after each stage of preprocessing of the documents and queries.

Though the code is designed to work on even bigger datasets it is advised to check if device has enough memory to be working on such huge data.

To change the model from Unigram to Bigram, you are requested to change the if statement on line 39 in main.py from true to false and vice versa