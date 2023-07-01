# Add your import statements here
from nltk.tokenize import punkt
from nltk.tokenize import treebank
from nltk.stem import PorterStemmer,WordNetLemmatizer 
from nltk.corpus import stopwords  
import re
import string as string_lib
import math
import numpy as np
from numpy import dot
from numpy.linalg import norm
import warnings
import operator
import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
warnings.filterwarnings("ignore")


from ctypes import *
from numpy.ctypeslib import ndpointer

so_file = '/home/lordguna/Desktop/gpuproject/parallelizedIR/sequential.so'
ccode = CDLL(so_file)

# Add any utility functions here
