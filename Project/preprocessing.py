from util import *

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		segmentedText = []
		# List of punctuations where we need to split the text
		punctuations = [';', '\.', '!', '\?']

		# Find all the indices where the text is to be split for sentence segmentation.
		indices = [m.end(0) for m in re.finditer(r'|'.join(punctuations), text)]

		# Split the text into segments using the indices calculated above
		start_index = 0
		stop_index = 0
		for index in indices:
			stop_index = index
			if len(text[start_index:stop_index].strip()) > 0:
				segmentedText.append(text[start_index:stop_index].strip())
			start_index = stop_index

		if len(text[start_index:]) > 0:
			segmentedText.append(text[start_index:].strip())

		return segmentedText




	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		tokenizer = punkt.PunktSentenceTokenizer()
		segmentedText = tokenizer.tokenize(text) 
		
		return segmentedText

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
		for sentence in text:
			tokenizedSentence = []
			for word in sentence.split(' '):
				if len(word.strip())>0: tokenizedSentence.append(word.strip())
			tokenizedText.append(tokenizedSentence)

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		tokenizer = treebank.TreebankWordTokenizer()
		for string in text:
			tokenizedText.append(tokenizer.tokenize(string))

		return tokenizedText

class InflectionReduction:

	def reduce(self, text,method='lemmatisation'):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []
		
		if (method=='lemmatisation'):
			lemmatizer = WordNetLemmatizer()
			for sentence in text:
				reducedSentence = []
				for word in sentence:
					reducedSentence.append(lemmatizer.lemmatize(word))
				reducedText.append(reducedSentence)

		else:
			ps = PorterStemmer() 

			for string in text:
				stemedtext = []
				for words in string:
					stemedtext.append(ps.stem(words))
				reducedText.append(stemedtext)

		
		return reducedText

class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []

		# stop_words = set(stopwords.words('english')).union(set(string_lib.punctuation))
		stop_words = set(stopwords.words('english'))
		
		for string in text:
			filteredtext = [word for word in string if not word in stop_words]
			stopwordRemovedText.append(filteredtext) 

		return stopwordRemovedText	