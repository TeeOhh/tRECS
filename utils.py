import pandas as pd 
import numpy as np
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import gensim
from gensim import corpora, models, similarities
from operator import itemgetter
import operator
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import spacy


def termdocument_matrix(data_frame):
	# ''' Description: This function takes in a dataframe of names and descriptions, and returns a doc/term frequency matrix
	# 	Parameters: a dataframe of names and descriptions
	# 	Output: m by n matrix where m=descriptions and n=words in descriptions '''
	
	
	descriptions = data_frame.iloc[:, 1].tolist()
	descriptions = [unicode(str(dscr), 'utf-8').split(" ") for dscr in descriptions]
	dictionary = gensim.corpora.Dictionary(descriptions)
	corpus = [dictionary.doc2bow(text) for text in descriptions]
	numpy_array = gensim.matutils.corpus2dense(corpus, num_terms=len(dictionary))
	return pd.DataFrame(numpy_array), dictionary.token2id

def make_binary(int):
	# '''Description: This function converts integers to either 0 if integer does not appear or 1 if integer appears
	# 	Parameters: integers (frequency of words)
	# 	Output: string of 1s and 0s'''
	if int > 0:
		return 1
	else:
		return 0

def get_num_of_topics(dataframe):
	# '''Description: This function takes in a dataframe of names and descriptions, and returns the optimum number of topics for use by gnesim's lsa model
	# 	Parameters: a dataframe of names and descriptions
	# 	Output: integer (optimum number of topics for lsa)'''
	frequency_matrix, bag_of_words = termdocument_matrix(dataframe)

	## Sort bag of words by their token2id ids
	sorted_bag_of_words = sorted(bag_of_words.items(), key=operator.itemgetter(1))

	## Set columns of matrix = bag of words
	frequency_matrix = frequency_matrix.T
	frequency_matrix.columns = [word[0] for word in sorted_bag_of_words]

	## Set indexes of matrix = course names
	frequency_matrix.index = dataframe[dataframe.columns[0]].tolist()
	
	# n = number of terms
	n=len(frequency_matrix.columns)
	# m = course names how many courses there are
	m=len(frequency_matrix.index)
	binary_matrix = frequency_matrix.applymap(make_binary)
	t=np.count_nonzero(binary_matrix)
	return m*n/t

def build_freq_matrix(dataframe):
	# '''Description: This function takes in a dataframe of names and descriptions and return matrix of number of documents a term occurs in
	# 	Parameters: a dataframe of names and descriptions
	# 	Output: m by 1 frequency matrix where m=number of documents where each word occurs'''
	frequency_matrix, bag_of_words = termdocument_matrix(dataframe)

	## Sort bag of words by their token2id ids
	sorted_bag_of_words = sorted(bag_of_words.items(), key=operator.itemgetter(1))

	## Set columns of matrix = bag of words
	frequency_matrix = frequency_matrix.T
	frequency_matrix.columns = [word[0] for word in sorted_bag_of_words]

	## Set indexes of matrix = course names
	frequency_matrix.index = dataframe[dataframe.columns[0]].tolist()
	
	## Make the frequency matrix a binary matrix for document frequency
	binary_matrix = frequency_matrix.applymap(make_binary)

	## Sum down and build word/document frequency matrix
	doc_frequency = binary_matrix.apply(sum).T
	doc_frequency = doc_frequency.to_frame().reset_index()

	doc_frequency = doc_frequency.rename(columns={'index' : 0, 0 : 1})
	doc_frequency.sort_values(by=1, inplace=True, ascending=False)
	return doc_frequency

def build_word_freq(dataframe):
	# '''Description: This function takes in a dataframe of names and descriptions and returns a matrix of each word's total number of occurences accross all documents
	# 	Parameters: a dataframe of names and descriptions
	# 	Output: m by 1 matrix where m=each word's frequency accross all documents'''
	word_freq_df = pd.DataFrame()

	## Build column of words and column of their corresponding occurences across all documents
	word_freq_df['Word'] = pd.Series(column for column in dataframe)
	word_freq_df['Frequency'] = pd.Series(sum(dataframe[column].values) for column in dataframe)

	## Set index as the words and sort by the occurences (most occurences at top and least at bottom)
	word_freq_df.set_index('Word', inplace=True)
	word_freq_df.sort_values(by='Frequency', inplace=True, ascending=False)
	
	return word_freq_df

## ------- START OF CLEANING METHODS -------

def master_clean(df, column, html, email, punc, non_ascii, stopwords, number, remove_nonenglish, stemorlem):
	if punc:
		df[column] = df[column].apply(remove_punc).to_frame()
	if html:
		df[column] = df[column].apply(remove_html).to_frame()
	if email:
		df[column] = df[column].apply(remove_email).to_frame()
	if non_ascii:
		df[column] = df[column].apply(remove_non_ascii).to_frame()
	if stopwords:
		df[column] = df[column].apply(remove_stop).to_frame()
	if number:
		df[column] = df[column].apply(remove_numbers).to_frame()
	if nonenglish:
		df[column] = df[column].apply(nonenglish).to_frame()
	if stemorlem == 'stem':
		df[column] = df[column].apply(stemmer).to_frame()
	elif stemorlem == 'lem':
		df[column] = df[column].apply(lemmatizer).to_frame()

	return df

def remove_punc(string):
	# '''Description: This function takes in a string of descriptions and return a tokenized string without punctuation
	# 	Parameters: String of descriptions
	# 	Output: Tokenized string with punctuation removed'''
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(string)
	return " ".join(tokens)

def remove_non_ascii(text):
	# '''Description: This function takes in a string of descriptions and return the string without non ASCII characters
	# 	Parameters: String of descriptions
	# 	Output: the string without non ASCII characters over 127'''
	stripped = (unicode(str(c), 'utf-8') for c in text if 0 < ord(c) < 127)
	return ''.join(stripped)

def remove_stop(string):
	# '''Description: This function takes in a string of descriptions and return the string without stopwords 
	# 	Parameters: String of descriptions
	# 	Output: String with stopwords removed (ex. a, and, the, etc)'''

	words = unicode(str(string), 'utf-8').split(" ")
	stop_words=[word.encode('utf-8') for word in stopwords.words('english')]
	filtered_words = [word for word in words if not word.lower() in stop_words]
	return " ".join(filtered_words)

def remove_email(text):
	# '''Description: This function takes in the string of descriptions and return the string without emails
	# 	Parameters: String of descriptions
	# 	Output: string with all emails removed'''

	match = re.compile('[^\s]+\\@(\\[?)[a-zA-Z0-9\\-\\.]+\\.([a-zA-Z]{2,3}|[0-9]{1,3})(\\]?)')
	return re.sub(match, '', text)

def remove_html(first_text):
	# '''Description: This function takes in the string of descriptions and return the string  with html code removed
	# 	Parameters: String of descriptions
	# 	Output: String with all html tags removed'''

	clean = re.compile('<.*?>')
	second_text = re.sub(clean, '', first_text)
	second_clean = re.compile('&nbsp;')
	
	return re.sub(second_clean, '', second_text)

def rebuild_freq_matrix(freq_matrix):
	# '''Description: This function takes in the document frequency matrix and return matrix with words that occur in only one document removed 
	#  	Parameters: Document frequency matrix
	# 	Output: Document frequency matrix with words that only occur in one document removed'''
	new_freq_matrix = freq_matrix.where(freq_matrix[1] != 1).dropna()

	return new_freq_matrix
	
def remove_occur_once(string, freq_matrix):
	# '''Description: This function takes in the document frequency matrix and return a string with words that occur in only one document removed"
	#  	Parameters: String of descriptions
	# 	Output: String with words removed that only occur in one document'''
	remove_words = freq_matrix.where(freq_matrix[1] != 1).dropna()[0].tolist()
	lis = unicode(str(string), 'utf-8').split(' ')
	words = [word for word in lis if word in remove_words]
	return " ".join(words)

def stemmer(text):
	# '''Description: This function takes in the string of descriptions and return string with all words stemmed
	# 	Parameters: String of descriptions
	# 	Output: String with all words stemmed (ex. "meeting" and "meetings" to "meeting")'''
	stemmer = PorterStemmer()
	lis = unicode(str(text), 'utf-8').split(" ")
	stemmed_words = [str(stemmer.stem(word)) for word in lis]

	return " ".join(stemmed_words)

def lemmatizer(text):
	# '''Description: This function takes in the string of descriptions and return string with all words lemmatized
	# 	Parameters: String of descriptions
	# 	Output: String with all words lemmatized (ex. "meeting" to "meeting" if noun and "meet" if verb)'''
	lemmatizer = WordNetLemmatizer()
	lis = unicode(str(text), 'utf-8').split(" ")
	lemm_words = [lemmatizer.lemmatize(word) for word in lis]
	return " ".join(lemm_words)

def remove_cutoff(text, keep_words):
	# '''Description: Thiis function takes in the string of descriptions and return string with only the words from the list of words to keep in the cuttoff slider
	# 	Parameters: String and a list of words to keep
	# 	Output: string of words with only words from list of words to keep'''
	lis = unicode(str(text), 'utf-8').split(" ")
	words = [word for word in lis if word in keep_words]
	if len(words) > 0:
		return " ".join(words)
	else:
		return ''

def remove_numbers(text):
	# '''Description: This function takes in the string of descriptions and return the string without numbers (useful for course syllabi)
	# 	Parameters: String of descriptions
	# 	Output: the string without numbers'''
	text = str(text)
	match = re.compile('\d')
	return re.sub(match, '', text)

def nonenglish(string):
	# '''Description: This function takes in the string of descriptions and return the string with nonenglish words removed (useful for course syllabi)
	# 	Parameters: String of descriptions
	# 	Output: the string with nonenglish words removed'''
	words = set(nltk.corpus.words.words())
	result=[w for w in nltk.wordpunct_tokenize(string) if w.lower() in words]
	return " ".join(result)


def ensemble(list_of_recommendations, amount):
	# '''Description: This function takes in the ranked list of recommendations from each model the user chooses (LSA, LDA, spacy, TF IDF) and output a ranked list of recommendations where the highest score from the sum of each model's ranked score is ranked as most similar.
	# 	Parameters: Ordered list of recommendations from LSA, LDA, spacy, and TF IDF
	# 	Output: List of recommendations in order'''
	d = {}
	for lis in list_of_recommendations:
		reverse=lis[::-1]
		newlist=list(enumerate(reverse, start=1))

		for pair in newlist:
			if pair[1] not in d:
				d[pair[1][0]] = pair[0]
			else:
				d[pair[1][0]] += pair[0]

	d = sorted(d.items(), key=operator.itemgetter(1))
	return [course for course in d][::-1][:amount]

def get_similar(course_name, amount, model_tuple):
	# '''Description: This function takes in a name and model ((LSA, TF IDF) and return a ranked dataframe where the first entry is the most similar description and the mth entry is the least similiar description, m=amount of descriptions the user chooses 
	# 	Parameters: name, amount, model_tuple
	# 	Output: dataframe of similar descriptions ranked by similarity where the first is most similar'''
	index = model_tuple[0]
	matrix = model_tuple[1]
	input_doc = matrix.loc[course_name]
	index.num_best = amount + 1
	top = index[input_doc]
	return get_names(matrix, top)

def get_similar_lda(course_name, amount, model_tuple):
	# '''Description: This function takes in a description and the LDA model returns a ranked dataframe where the first entry is the most similar description and the mth entry is the least similiar description, m=amount of descriptions the user chooses 
	# 	Parameters: name, amount, model_tuple
	# 	Output: dataframe of similar descriptions ranked by similarity where the first is most similar'''
	index = model_tuple[0]
	matrix = model_tuple[1]
	input_doc = matrix.loc[course_name]
	index.num_best = amount + 1
	top = index[input_doc]
	top = top[0]
	return get_names(matrix, top)

def get_similar_spacy(course_name, amount, df):
	# '''Description: The function takes in a description and the dataframe of course name and the spacy model returns a ranked dataframe where the first entry is the most similar description and the mth entry is the least similiar description, m=amount of descriptions the user chooses 
	# 	Parameters: name, amount, data frame of names
	# 	Output: dataframe of similar descriptions ranked by similarity where the first is most similar'''
	input_doc= df[course_name]
	top = {k: input_doc.similarity(v) for k, v in df.items() if len(v) > 0}
	top = sorted(top.items(), key=operator.itemgetter(1), reverse = True)
	top = top[:amount + 1]
	return top

def get_names(df, lis):
	# '''Description: This function takes in a dataframe/list from the get_similar function and returns a dataframe where similarities are indexed by the course name
	# 	Parameters: dataframe of similarity score, list of names
	# 	Output: dataframe where similarities are indexed by the course name'''
	
	named = [(df.index[couple[0]], couple[1]) for couple in lis]

	return named


def concat_descr(column):
	# '''Description: This function takes in the columns from the dataframe and concatenates them for the spacy model
	# 	Parameters: 2+ decription columns from dataframe
	# 	Output: concatenated string of descriptions '''
	dscr = ''
	for d in column:
		d = d.decode('utf-8')
		dscr += d
		dscr += ' '

	return dscr

def get_all_entities(document):
	# '''Description: This function converts a document to a dictionary of entity labels and a list of the entities (of that label) found in the document
	# 	Parameters: String of descriptions
	# 	Output: A dictionary in form: entity label: [word1, word2, ....]'''
	nlp = spacy.load('en')
	entities = []
	doc = nlp(document)
	d = {}
	for ent in doc.ents:
		label = ent.label_
		text = ent.text
		if label in d:
			d[label].append(text)
		else:
			d[label] = [text]
	return d

def build_ent_dic(entity_dic):
	# '''Description: This function takes a dictionary of entities and their words found in the data set and converts it to a new dictionary of dictionaries that contains the amount of times each word occurs in the dataset, for each category of entity
	# 	Params: Dictionary of form: entity : [word1, word2, ...]
	# 	Output: Dictionary of dictionaries in form: Entity label : { Word1 : occur, word2: occur }'''

	d = {}
	for label, texts in entity_dic.items():
		if label not in d:
			d[label] = {}
		for word in texts:
			if word not in d[label]:
				d[label][word] = 1
			else:
				d[label][word] += 1
	return d

def breakdown_ents(label, d):
	# '''Description: This funcitons takes in a category label of entity and returns the data for a pie chart that breaks down the words of that category of entity.
	# 	Parameters: The label of entity and the dictionary of entity occurence counts from the above function
	# 	Output: The data needed for the pie chart'''

	dic = d[label]
	x = [word for word, amount in dic.items()]
	y = [amount for word, amount in dic.items()]
	pie_data = [go.Pie(labels=x, values=y, textinfo='percent', name=label)]
	
	return pie_data

def get_dataframe(label, d):
	# '''Description: This functions builds a dataframe of words and their occurence counts for a particular entity
	# 	Parameters: The label of entity and the dictionary of entity occurence counts from the build_ent_dic function
	# 	Output: A dataframe of words and their occurence counts for the desired entity'''

	df = pd.DataFrame(d)
	df = df.T
	df = df.loc[label].dropna().to_frame()
	df = df.reset_index()
	df = df.rename(columns={"index": "Entities", label: "Counts",})
	df = df.sort_values(by='Counts', ascending=False)
	return df