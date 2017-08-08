import utils
import pandas as pd
import lda
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import plotly.graph_objs as go
from sklearn.manifold import TSNE



class lda_cluster_graph:

	# '''
	#   description: Creates a scatter plot of documents, colored by the topic that is most prevalent within them.
	#   contains: dim: dimension of graph
	#             threshold: minimum percentage of the maximum topic in a document which can be included in a "cluster 
	#             titles_original: titles of objects in  in original data set
	#             descriptions_original: descriptions of objects in  in original data set
	#             titles_current/descriptions_current: title/descriptions of object after filtering with threshild
	#             nt: number of topics currently in lda model
	#             lda_model: lda model in use
	#             X_topics_current/original: document by topic matrix
	# '''
	
	def __init__(self, df):
		self.dim ="2d"
		self.threshold = 0
		self.titles_original =  np.array(df.iloc[:,0].tolist())
		self.descriptions= df.iloc[:,0].tolist()

	def make_lda(self, nt, iterations):
		# '''
		#   description: sets important attributes and creates lda model
		#   params:     nt-number of topics for lda
		#               iterations: number of iterations for lda
		#               dim: 2d or 3d grpah
		#               threshold: minimum percentage of the maximum topic in a document which can be included in a "cluster"
		# '''
		
		self.nt = nt        
		

		self.cvectorizer = CountVectorizer(min_df=5, stop_words='english')
		cvz = self.cvectorizer.fit_transform(self.descriptions)

		# train an LDA model
		self.lda_model = lda.LDA(n_topics=nt, n_iter=iterations)
		self.X_topics_original = self.lda_model.fit_transform(cvz)

		#initialize current stuff
		self.X_topics_current = self.X_topics_original
		self.titles_current = self.titles_original

	def set_graph(self, new_threshold, new_dim):
		# '''
		# desc: create plot 
		# params: new threshold value, new graph dimension
		# returns:plotly figure'''
		if new_dim != self.dim:
			self.dim = new_dim
		if new_threshold != self.threshold:
			self.set_threshold(new_threshold)
		return self.create_plot()


	def make_legend(self):
		# '''
		#   desc: creates a dictionary where the index is the top words for a topic and the entry is the color assigned to that topic
		#   returns: legend in form of a dictionary

		colormap = self.make_colormap()
		topic_summaries = self.get_topicwords()
		legend = {topic_summaries[i]:colormap[i] for i in range(self.nt)}
		return legend

	# User Interaction

	def set_threshold(self, new_threshold):
		# '''
		#   desc: reset the threshold ane recalcualte x_topics
		#   params: new threshold  
		# '''
		if new_threshold > self.threshold:
			self.titles_current = self.titles_original
			self.X_topics_current = self.X_topics_original

		self.threshold = new_threshold
		_idx = np.amax(self.X_topics_current, axis=1) > self.threshold  # idx of doc that above the threshold
		self.X_topics_current = self.X_topics_current[_idx] #this code removes docs that don't have a clearly defined topic
		self.titles_current = self.titles_current[_idx] #also remove corresponding titles
		
	
	#Create Plotly Stuff

	def create_plot(self):
	# '''
	#   description: create plotly figure
	#   returns:plotly figure
	# '''
		colormap = self.make_colormap()
		_lda_keys = self.get_lda_keys()
		
		if self.dim =="2d":
			# reduce the dimesnsions of X_topics
			# angle value close to 1 means sacrificing accuracy for speed
			# pca initializtion usually leads to better results 
			tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')

			# 20-D -> 2-D
			tsne_lda = tsne_model.fit_transform(self.X_topics_current)

			#create tracem
			trace1 = go.Scatter(
			x = tsne_lda[:, 0],
			y = tsne_lda[:, 1],
			mode = 'markers',
			marker =dict(color = colormap[_lda_keys]),
			text= self.titles_current
			)

			data = [trace1]
			layout = go.Layout(xaxis = dict(visible = False),yaxis = dict(visible = False))
			fig2d = go.Figure(data=data, layout = layout)
			return fig2d

		#same but with 3d graph 
		else:
			tsne_model = TSNE(n_components=3, verbose=1, random_state=0, angle=.99, init='pca')

			# 20-D -> 3-D
			tsne_lda = tsne_model.fit_transform(self.X_topics_current)
			trace1 = go.Scatter3d(
				x = tsne_lda[:, 0],
				y = tsne_lda[:, 1],
				z =  tsne_lda[:, 2],
				mode = 'markers',
				marker = dict(color = colormap[_lda_keys]),
				text = self.titles_current
			)

			data = [trace1]
			layout = go.Layout(scene = dict(xaxis = dict(visible = False),yaxis = dict(visible = False), zaxis = dict(visible = False) ))
			fig3d = go.Figure(data=data, layout = layout)
			return fig3d

	def get_lda_keys(self):
		# '''
		#   desc: finds most prevalent topic in each document
		#   returns: list with index of most prevalent topic
		# '''
		_lda_keys = []
		for i in xrange(self.X_topics_current.shape[0]):
			_lda_keys +=  self.X_topics_current[i].argmax(),
		return _lda_keys

	def make_colormap(self):
		max_value = 16581375 #255**3
		interval = int(max_value / self.nt)
		colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
		colormap = ["rgb" + str((int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16))) for i in colors]
		
		return np.array(colormap)

	# Other 
	

	def get_topicwords(self):
		# '''
		 #  desc: find top words for each topic
		 #  returns: list of top words
		# '''
		n_top_words = 3 # number of keywords we show
		topic_summaries = []
		topic_word = self.lda_model.topic_word_  # all topic words
		vocab = self.cvectorizer.get_feature_names()
		for i, topic_dist in enumerate(topic_word):
			topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1] # get!
			topic_summaries.append(' '.join(topic_words)) # append!
		return topic_summaries




def make_number_line(similar, group, df):
	# '''
	#   description: makes a scatter plot of most relevant courses can be grouped by group data
	#   params: similar: list of tuples, where each tuple is a course name and its similarity score.
	#           group: string, name of column containing group data
	#           df: data frame where the row index is by course title and column name is the string group. contains the name of the group the course belongs to
	#   returns: plotly figure
	# '''
	titles, scores = zip(*similar)
	if group == None:
		y_axis = [0]*len(scores)
		layout = dict(yaxis = dict(visible = False), title = 'Similar Courses')
	else:
		y_axis = [df.loc[title][group] for title in titles]
		layout = dict(title = 'Similar Courses')

	trace = go.Scatter(
	x = scores,
	y= y_axis,
	mode = 'markers',
	text = titles)

	data = [trace]
	fig = go.Figure(data=data, layout = layout)
	return fig