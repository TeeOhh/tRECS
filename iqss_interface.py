## Recommendation interface takes in user inputted data and cleans it as required,
## and then builds chosen models and does ensemble document similarity


# Requirements 
import pandas as pd
import utils
import numpy as np
import utils
import gensim
from gensim import corpora, models, similarities
from operator import itemgetter
import spacy

class iqss_interface:
	## Description: This object holds a dataframe w/o cleaning and w/ cleaning
	## Contains: Methods for loading data, cleaning data, etc.
	
	def __init__(self):

		self.df = pd.DataFrame()
		self.clean_df = pd.DataFrame()
		self.freq_matrix = pd.DataFrame()
		self.freq_matrix_keep = pd.DataFrame()

		# Model variables
		self.iqss_model = model_object()

#Step 1: Initialize Data Frame

	def load_data(self, file_name):
	## Description: Saves data as pandas data frame
	## Params: Name of CSV containing data
	## Returns: Nothing

		self.df = pd.read_csv(file_name)
		self.df = self.df.drop_duplicates(keep='first')
		self.df = self.df.dropna()


	def load_columns(self, names, desc):
	## Description: Resaves data frame to only include these columns
	## Params: Name of column with names and name of columns with descriptions
	## Returns: Nothing

		## Re-order with just first two columns as names, description
		self.df = self.df[[names, desc]]
		self.clean_df = self.df.copy()

#Step 2: Build Document Frequency Matrix
	
	def build_doc_freq(self):
		self.freq_matrix = utils.build_freq_matrix(self.clean_df)

#Step 3: Clean Data

	def default_clean(self):
		## Description: Select second column (column w/ descriptions) and do the default cleaning
		## of removing punctuation and non-ascii characters
		## Params: None
		## Returns: Nothing

		self.clean_df[self.clean_df.columns[1]] = self.clean_df[self.clean_df.columns[1]].apply(utils.remove_non_ascii).to_frame()
		self.clean_df[self.clean_df.columns[1]] = self.clean_df[self.clean_df.columns[1]].apply(utils.remove_punc).to_frame()
	
	def stem_lem_clean(self, stem_or_lem):
		## Description: If user chooses stem or lemmatization select second column (column w/ descriptions) 
		## and stem/lemmatize the data
		## Params: A string of 'stem' or 'lemma'
		## Returns: Nothing

		if stem_or_lem == 'stem':
			self.clean_df[self.clean_df.columns[1]] = self.clean_df[self.clean_df.columns[1]].apply(utils.stemmer).to_frame()
		elif stem_or_lem == 'lemma':
			self.clean_df[self.clean_df.columns[1]] = self.clean_df[self.clean_df.columns[1]].apply(utils.lemmatizer).to_frame()

	def additional_clean(self, list_of_options):
		## Description: Run through list of options chosen and do each of the cleaning steps chosen by user
		## Params: List of additional cleaning options
		## Returns: Nothing

		if u'html' in list_of_options:
			self.clean_df[self.clean_df.columns[1]] = self.clean_df[self.clean_df.columns[1]].apply(utils.remove_html).to_frame()
		self.default_clean()
		if u'nonenglish' in list_of_options:
			self.clean_df[self.clean_df.columns[1]] = self.clean_df[self.clean_df.columns[1]].apply(utils.nonenglish).to_frame()
		if u'numbers' in list_of_options:
			self.clean_df[self.clean_df.columns[1]] = self.clean_df[self.clean_df.columns[1]].apply(utils.remove_numbers).to_frame()
		if u'stop' in list_of_options:
			self.clean_df[self.clean_df.columns[1]] = self.clean_df[self.clean_df.columns[1]].apply(utils.remove_stop).to_frame()
		if u'lower' in list_of_options:
			self.clean_df[self.clean_df.columns[1]] = self.clean_df[self.clean_df.columns[1]].apply(unicode.lower).to_frame()
		if u'once' in list_of_options:
			self.freq_matrix = utils.rebuild_freq_matrix(self.freq_matrix)
			self.clean_df[self.clean_df.columns[1]] = self.clean_df[self.clean_df.columns[1]].apply(lambda x: utils.remove_occur_once(x, self.freq_matrix)).to_frame()
		
	def remove_cutoff(self):
		## Description: Uses a list of words that the user wishes to keep from the slider cut-off
		## then removes the words not in that list from the clean dataframe
		## Params: None
		## Returns: Nothing

		keep_words = self.freq_matrix_keep[0].tolist()
		self.clean_df[self.clean_df.columns[1]] = \
		self.clean_df[self.clean_df.columns[1]].apply(lambda x: utils.remove_cutoff(x, keep_words))

#Step 4: Build Models

	def build_models(self, tfidf_bool, lsa_bool, lsa_nt, lda_bool, lda_nt, spacy_bool):
		## Description: Builds the models (TFIDF, LSA, LDA, Spacy)
		## Params: If user wants to build chosen model (model_bool = 1), nt: number of topics
		## Returns: Nothing

		self.iqss_model.build_models(tfidf_bool, 
									lsa_bool, lsa_nt, 
											lda_bool, 
											lda_nt,
											spacy_bool, 
										self.clean_df)

	def load_ents(self):
		descriptions = self.df[self.df.columns[1]]
		all_dscr = utils.concat_descr(descriptions)
		self.all_entities = utils.get_all_entities(all_dscr)
		self.entity_dic = utils.build_ent_dic(self.all_entities)


#Step 5: Get similar documents

class model_object:
# Description: Takes dataframe and creates models/similarity
# Contains: Methods to build models and conduct document similarity	

	def __init__(self):

		self.df = pd.DataFrame()
		self.tfidf = tuple()
		self.lda = tuple()
		self.lsa = tuple()
		self.spacy = tuple()

	#BUILDING TOOLS

	def build_base(self, df):
		# Description: Initialize data variables
		self.df = df
		## Description: Build and save dictionary, corpus, and BOW(Bag-of-words) matrix
		## Params: Dataframe
		## Returns: Dictionary, Corpus

		descriptions = df.iloc[:,1].tolist()
		descriptions = [str(course_dscr).split(" ") for course_dscr in descriptions]
		dictionary = gensim.corpora.Dictionary(descriptions)
		corpus = [dictionary.doc2bow(text) for text in descriptions]
		return dictionary, corpus, self.build_bows(dictionary)
		

	def build_bows(self, dictionary):
		## Description: Build BOW vectors for each description in data
		## Params: Dictionary
		## Returns: BOW matrix

		bow_matrix = self.df.iloc[:, :2].copy()
		for row in bow_matrix.index:
			desc = bow_matrix.loc[row][1].split(" ")
			bow = dictionary.doc2bow(desc)
			bow_matrix.loc[row][1]= bow
		bow_matrix = bow_matrix.set_index([bow_matrix.columns[0]])
		self.bow_matrix = bow_matrix
		return bow_matrix


	def build_tfidf_base(self, corpus, bow_matrix):
		## Description: Build and save objects common to TFIDF and LSA
		## Params: Corpus, BOW matrix
		## Returns: TF-IDF corpus and matrix

		tfidf_model = models.TfidfModel(corpus)
		tfidf_corpus= tfidf_model[corpus]
		tfidf_matrix = bow_matrix.apply(lambda x: tfidf_model[x[0]], 1)
		return tfidf_corpus, tfidf_matrix
		

	#MODEL OBJECTS
	#A model object consists of gensim similarity index and matrix containing transformed data

	def build_tfidf(self, tfidf_corpus):	    
		## Description: Builds TFIDF and does Gensim similarity
		## Params: TFIDF corpus
		## Returns: Similarities suggested by the model	 

		index = similarities.MatrixSimilarity(tfidf_corpus)
		return index

	
	def build_lsa(self, nt, dictionary, tfidf_corpus, tfidf_matrix):
		## Description: Builds LSA model and performs document similarity
		## Params: Number of topics, dict, TFIDF corpus, TFIDF matrix
		## Returns: Similarity index and matrix

		lsa_model = models.LsiModel(tfidf_corpus, id2word= dictionary, num_topics=nt)
		index = similarities.MatrixSimilarity(lsa_model[tfidf_corpus])
		matrix = tfidf_matrix.apply(lambda x: lsa_model[x], 1)
		return (index, matrix)
		
	
	def build_lda(self, nt, corpus, dictionary, bow_matrix):
		## Description: Builds LDA and does document similarity
		## Params: Number of topics, corpus, dict, BOW matrix
		## Returns: Similarity index and matrix

		lda_model = models.LdaModel(corpus, id2word= dictionary, num_topics=nt)
		self.lda_model = lda_model
		index = similarities.MatrixSimilarity(lda_model[corpus])
		matrix = bow_matrix.apply(lambda x: [lda_model[x[0]]], 1)
		return (index, matrix)

	def build_spacy(self, df):
		## Description: Builds spacy similarity model
		## Params: dataframe
		## Returns: similar documents

		nlp = spacy.load('en')
		docs = df.set_index([df.columns[0]])
		docs = docs.iloc[:,0].to_dict()
		func = lambda x: nlp(x.decode('utf-8'))
		docs = {k: func(v) for k, v in docs.items()}
		return docs


	#Build models of the user's choice

	def build_models(self, tfidf_bool, lsa_bool, lsa_nt, lda_bool, lda_nt, spacy_bool, df):
		## Description: Build common variables
		## Params: If chosen by user ((TFIDF, LSA, LDA, spacy)_bool = 1), nt: number of topics
		## Returns: Nothing

		dictionary, corpus, bow_matrix = self.build_base(df)

		if tfidf_bool or lsa_bool:
			tfidf_corpus, tfidf_matrix = self.build_tfidf_base(corpus, bow_matrix)
			self.tfidf_matrix = tfidf_matrix
			if tfidf_bool:
				self.tfidf = (self.build_tfidf(tfidf_corpus), tfidf_matrix)
			if lsa_bool:
				self.lsa = self.build_lsa(lsa_nt, dictionary, tfidf_corpus, tfidf_matrix)
		if lda_bool:
			self.lda = self.build_lda(lda_nt, corpus, dictionary, bow_matrix)
		if spacy_bool:
			self.spacy = self.build_spacy(df)

	def get_similar(self, course_name, amount):
		 ## Description: Takes in courses ranked as similar by TF-IDF, LDA, LSA, and Spacy
		 ## and using the ensemble method, gives out similar ones based off the 
		 ## chosen model outputs
		 ## Params: Name of the course and amount of simliar output
		 ## Returns: List of ranked similar courses

		lis = [self.tfidf, self.lda, self.lsa, self.spacy]
		similar_lis = []
		amount = int(amount)
		for model in lis:
			if len(model) > 0:
				if model == self.lda:
					courses = utils.get_similar_lda(course_name, amount, model)
				elif model == self.spacy:
					courses = utils.get_similar_spacy(course_name, amount, self.spacy)
				else:
					courses = utils.get_similar(course_name, amount, model)
				
				courses = [tup for tup in courses if course_name != tup[0]]
				similar_lis.append(courses)
			else:
				similar_lis.append([])

		return utils.ensemble(similar_lis, amount)

	










