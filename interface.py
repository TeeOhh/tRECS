# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, Event, State
import plotly.graph_objs as graph
import pickle
import pandas as pd
import functools32
import iqss_interface
import utils
import os
from visualizations import lda_cluster_graph as lda_cluster

'''
	Description: This app is designed to allow users to do in-depth preprocessing and cleaning on their language datasets, 
				 then build different statistical NLP methods such as TF-IDF, LSA, LDA, and SpaCY (gloVe vectors), then 
				 test a similarity/recommendation system, and finally analyze the dataset and models by visualizing the 
				 different entities, topic clustering, etc. The idea is that this gives a user a high level application 
				 that does all of this within a GUI rather than writing code.

	Pages: 
		Uploading page - User uploads data and selected the necessary columns
		Cleaning page - User selects the different cleaning options we provide
		Cleaning page (2) - User can select thresholds to remove words that are not seen in a certain amount of documents.
		Model building page - User can select the desired statistical models to be built
		Testing page - User can test the models using a recommendation system, entering a document name and getting a certain
		               amount of recommended (similar) documents
		Analyzing page - User can select to visualize the different categories of entities within their data and 
		                 visualize the clusters of topics within their data.

'''

app = dash.Dash()
app.config.supress_callback_exceptions=True

## Load current directory files for uploading of matrix
## Get current directory
cd = os.getcwd()
## Create list of files in current directory
files = list(os.listdir(cd))

# ----- LOAD EXTERNAL STYLING -----

my_css_url = "https://codepen.io/anon/pen/bRxMvz.css"

app.css.append_css({
	"external_url": my_css_url
})

## keep track of current step so slide can be updated when button is pressed
step = 0

## build instance of the backend object
interface_obj = iqss_interface.iqss_interface()


## ----- START OF CONTENT -----

app.layout = html.Div([
	## Container with top 'progress bar' and next button
	
	html.Div([
		html.P('VPAL RECOMMENDATION SYSTEM DEVELOPER')
		], id='header'),
	html.Div([
	html.Div([
		dcc.Slider(
			id='slider',
			min = 0,
			max = 5,
			marks = ['Load Data', 'Clean', 'Clean (part 2)', 'Build Model(s)', 'Test', 'Analyze'],
			value = 0,
			disabled='true',
		)
	], className='column', id='top-slider-container'),

	html.Button('Next Step', id='next-button', className = 'next-btn-invis'),
	
	## Content place holder for each page ('step') of the process

	html.Div(id='content')], className = 'container')   
])


# ----- START OF INTERACTIVITY -----

@app.callback(
	## When slider is changed, check the global step variable and display the necessary content in the main container
	Output(component_id='content', component_property='children'),
	[Input(component_id='slider', component_property='value')]
	)

def test(slider_value):
	## ----- Load Data Page -----
	# '''
	# 	Description: This page allows the user to choose from a list of .csv files in the current directory to upload.
	# 				 The user can then choose the desired columns to run through the interface (label column, and data column).
	# 				 The user will see a live preview of the uploaded dataframe when both the label and data column are chosen.
	# '''

	if slider_value == 0:
		return (
			html.Div(['Upload your data and select the columns and you \
				will see a preview of your data here...'], id = 'preview-df'),
			html.Div([
				dcc.Dropdown(
					id = 'file-chooser',
					options = [
						{'label' : file, 'value' : file} for file in files if '.csv' in file
					],
					placeholder = 'Choose file...'
				)
			], id='file-choosing-container'),
			html.Div([
				html.P('LOAD DATA'),
				html.Button('Upload', id='upload-btn')
			], className = 'section-container-s'),
			html.Div([
				html.P('CHOOSE COLUMNS'),
				html.Div([
					html.P('Label Column'),
					dcc.Dropdown(
						options=[],
						id='label-col'
						)], className = 'half-container'),
				html.Div([
					html.P('Data Column'),
					dcc.Dropdown(
						options=[],
						id='data-col'
						)], className = 'half-container')
				], className = 'section-container-m'),
			)
	
	## ----- Clean Data Page -----
	# '''
	# 	Description: This page allows the user to select a number of cleaning steps to clean their data.
	# 	Default cleaning steps: Removing non-ascii and removing punctuation.
	# 	Optional cleaning steps: Removing HTML, remove words that occur in only one document, lowercase all words,
	# 							  remove all 'stop' words, remove all numbers, remove all non-english words, and stem or lem all words.
	# '''

	elif slider_value == 1:
		global interface_obj
		## Build document frequency matrix from clean_df for slider functionality
		interface_obj.default_clean()
		interface_obj.build_doc_freq()

		return (
			html.Div(['If you choose to select additional options you will \
				see a preview of your data after being cleaned here...'], id='clean-preview'),
			
			html.Div([
				html.P('DEFAULT OPTIONS'),
				dcc.Checklist(
					options = [
						{'label' : 'Remove non-ascii', 'value' : 'ascii', 'disabled' : 'true'},
						{'label' : 'Remove punctuation', 'value' : 'punctuation', 'disabled' : 'true'},
						],
					id= 'default-options',
					values = ['ascii', 'punctuation'],
					labelClassName = 'cleaning-options'
					)
			], className = 'default-container'),
			
			html.Div([
				html.P('ADDITIONAL OPTIONS'),
				html.Div([
					dcc.Dropdown(
						options = [
						{'label' : 'Remove HTML', 'value' : 'html'},
						{'label' : 'Remove words that only occur in one document', 'value' : 'once'},
						{'label' : 'Lowercase', 'value' : 'lower'},
						{'label' : 'Remove stop words', 'value' : 'stop'},
						{'label' : 'Remove all numbers', 'value' : 'numbers'},
						{'label' : 'Remove all nonenglish words', 'value' : 'nonenglish'},
						],
						multi=True,
						id='add-options'),
					], className = 'half-container'),

				html.Div([
					dcc.Dropdown(
						options = [
							{'label' : 'Stem', 'value' : 'stem'},
							{'label' : 'Lemmatize', 'value' : 'lemma'}
							],
						id='stem-lem-options'
						)
					], className = 'half-container'),
				 ]),
		)
	
	## ----- Cutoff Page -----
	# '''
	# 	Description: This page allows the user to cutoff words that occur below or above a certain threshold of documents.
	# 				 (The standard % seen in research for this is below 1%)
	# '''

	elif slider_value == 2:
		## Rebuild document frequency matrix after finishing cleaning
		global interface_obj
		interface_obj.build_doc_freq()

		return html.Div([
				html.Div(id='download-confirmation'),
				html.Button('DOWNLOAD DATA', id='download-btn', style={'marginBottom' : '3%'}),
				html.Div([], id='cutoff-confirmation'),
				html.Div([
					html.P('Bottom Threshold'),
					html.Div(id='bottom-thresh', className='cutoff_words')],
					className='cutoff_container', style={'paddingRight' : '5%'}),
				
			## --- GRAPH AND THRESHOLD CUTOFF CONTAINER --- 
			html.Div([
				html.P('CUT OFF THRESHOLDS (Optional)',
					),

				## Graph of words and document frequency: x = doc. freq., y = ordered list of words
				dcc.Graph(
					id='graph-with-slider', 
					animate=True,
					## Set data: x = doc. freq., y = index of dataframe (ordered word IDs)
					figure={'data': []},
					),
				
				dcc.RangeSlider(
					id='percent-slider',
					min=0,
					max=20,
					step=.2,
					marks=[int(x) for x in range(0, 101, 5)],
					value = [0, 20]),

			], id='graph-container'),

			html.Div([
				html.P('Top Threshold'),
				html.Div(id='top-thresh', className='cutoff_words')],
					className='cutoff_container', style={'paddingLeft' : '5%'}),

			html.Div([
				html.Button('CUTOFF', id='cutoff-btn')
				], style={'clear' : 'both'})        
			], id='cutoff_main_container')
	
	## ----- Build Model Page -----
	# '''
	# 	Description: This page allows the user to select and build a number of statistical language models for their data. 
	# 	Model options: TF-IDF, LDA, LSA, SpaCy
	# '''

	elif slider_value == 3:
		return (html.Div([
				html.Div([], id='build-confirmation'),
				html.Div([
					html.P('BUILD MODEL(S)')
					], className='section-container-s'),
				dcc.Checklist(
					options = [
						{'label' : 'Tf-Idf', 'value' : 'tf_idf'},
						{'label' : 'LSA', 'value' : 'lsa'},
						{'label' : 'LDA', 'value' : 'lda'},
						{'label' : 'Spacy', 'value' : 'spacy'},
					], 
					values = [],
					id='model-choices',
					labelClassName = 'model_choices'
					),
			]),
				
				html.Div([
					html.Div([
						html.P('tf–idf, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.')
						], className='fourth-container'),
					html.Div([
						html.P('Latent semantic analysis (LSA) is a technique in natural language processing, in particular distributional semantics, of analyzing relationships between a set of documents and the terms.')
						], className='fourth-container'),
					html.Div([
						html.P('latent Dirichlet allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar.')
						], className='fourth-container'),
					html.Div([
						html.P('spaCy excels at large-scale information extraction tasks. It\'s written from the ground up in carefully memory-managed Cython. Independent research has confirmed that spaCy is the fastest.')
						], className='fourth-container')
				]),

				html.Div([
					html.Button('BUILD', id='build-btn')
					], style={'clear' : 'both'}, className='section-container-m')
		)

	## ----- Test Page -----
	# '''
	# 	Description: This page allows the user to select a label (course name) from the list of data from the label column they chose when uploading.
	# 	             From here, the user then enters the number of similar labels (courses) they wish to output when ran.
	# 	             The user then presses the run button and the system outputs the top X amount of similar labels.
	# '''

	elif slider_value == 4:
		global interface_obj
		courses = interface_obj.clean_df.iloc[:, 0]
		return html.Div([
				html.Div([
					dcc.Input(
						placeholder = 'Enter amount...',
						type='number',
						id = 'amount-entry'
						),
					html.Button('RUN', id='get-similar-btn')
					]),
				html.Div([
					html.Div([
						html.P('CHOOSE A COURSE', className = 'test-headers'),
						dcc.Dropdown(
							options = [
								{'label' : course_name, 'value' : course_name} for course_name in courses
							],
							id = 'course-selection'
							)
						], className = 'half-container'),
					html.Div([
						html.P('SIMILAR COURSES', className = 'test-headers'),
						html.Ul(
							children = [], 
							id='similar-courses', 
							style={'listStyleType' : 'none'})
						], id = 'course-recom-container')
					], id = 'testing-container')
			])

	## ----- Analyze Page -----
	# '''
	# 	Description: This page allows the user to select two different methods of analyzing their data.
	# 	Analyzing options: Detecting entities (using SpaCY) and analyze the topic models of LDA.
	# '''

	elif slider_value == 5:
		return html.Div([
			html.Div([
			html.P('ANALYZING OPTIONS'),
			dcc.Dropdown(
				options = [
					{'label' : 'Entities', 'value' : 'ents'},
					{'label' : 'Visualize Topics', 'value' : 'topics'}
				],
				id = 'analyze-dropdown'
				)], id='analyze-options'),
			html.Div(id='analyze-content')
			])


## ------- METHODS FOR APP FLOW AND BACKEND MANIPULATION -------

def upload_file(file_name):
	# '''
	# 	Description: Uploads the user selected file name and builds the dataframe in the backend object.
	# 	Params: User selected filename from the dropdown.
	# 	Returns: List of columns that exist in the csv file.
	# '''

	global interface_obj
	interface_obj.load_data(file_name)
	return interface_obj.df.columns.tolist()

def update_table(df, columns):
	# '''
	# 	Description: Builds an html table from the desired dataframe and columns.
	# 	Params: Dataframe and the list of necessary column names
	# 	Returns: Table sample of first few entries in dataframe
	# '''

	max_amount = len(df)
	## If the data does not have 3 or more entries, display only one entry as a sample
	if max_amount < 3:
		return html.Table(
			[html.Tr([html.Th(col) for col in columns])] +

			[html.Tr([
				html.Td(df.iloc[i][col][:150] + ' ...') for col in columns
				]) for i in range(1)]
			)
	## If data has more than 3 entries, display first 3 entries as a sample
	else:
		return html.Table(
				[html.Tr([html.Th(col) for col in columns])] +

				[html.Tr([
					html.Td(df.iloc[i][col][:150] + ' ...') for col in columns
					]) for i in range(3)]
				)

## ------- WHEN USER PRESSES NEXT STEP, CHANGE CONTENT AND INCREMENT SLIDER --------
@app.callback(
	Output(component_id='slider', component_property='value'),
	events=[Event('next-button', 'click')]
	)

def next():
	# '''
	# 	Description: Used for switching of page content. When next step button pressed, increments or resets global variable step.
	# 	Params: None
	# 	Returns: Incremented or reset global step variable.
	# '''

	global step
	if step == 5:
		step = 0
		return step
	else:
		step += 1
		return step

## ------- LOAD DROPDOWNS WITH COLUMN OPTIONS -------
@app.callback(
	Output('label-col', 'options'),
	events=[Event('upload-btn', 'click')],
	state=[State('file-chooser', 'value')]
	)
def load_label_drop(file_name):
	# '''
	# 	Description: Loads the label dropdown selector with all of the column options from the uploaded dataframe.
	# 	Params: The file name of the csv file.
	# 	Returns: Dropdown values & labels to fill the dash dropdown selector.
	# '''

	columns = upload_file(file_name)
	return [{'label' : column, 'value' : column} for column in columns]

@app.callback(
	Output('data-col', 'options'),
	events=[Event('upload-btn', 'click')],
	state=[State('file-chooser', 'value')]
	)
def load_data_drop(file_name):
	# '''
	# 	Description: Loads the label dropdown selector with all of the column options from the uploaded dataframe.
	# 	Params: The file name of the csv file.
	# 	Returns: Dropdown values & labels to fill the dash dropdown selector.
	# '''

	columns = upload_file(file_name)
	return [{'label' : column, 'value' : column} for column in columns]

## ------- UPDATE LIVE DATAFRAME PREVIEW WITH SELECTED COLUMNS -------
@app.callback(
	Output('preview-df', 'children'),
	[Input('label-col', 'value'), Input('data-col', 'value')]
	)

def reduce_df(column1, column2):
	# '''
	# 	Description: When user selects options for label and data column, load the dataframe preview with the current selected columns.
	# 	Params: Column name of label column and column name of data column.
	# 	Returns: Confirmation that the users dataframe has been uploaded and shows the dataframe preview returned from update_table() method.
	# '''
	col1 = str(column1)
	col2 = str(column2)
	if col1 !='None' and col2 != 'None':
		global interface_obj
		interface_obj.load_columns(col1, col2)
		return (
			html.P('DATAFRAME PREVIEW'), 
			update_table(interface_obj.df, [col1, col2]))
	else:
		return 'Upload your data and select the desired columns and you \
				will see a preview of your data here...'


## ------- HIDE NEXT BUTTON UNTIL USER UPLOADS DATA -------
@app.callback(
	Output('next-button', 'className'),
	[Input('label-col', 'value'), Input('data-col', 'value')]
	)

def show_next_button(column1, column2):
	# '''
	# 	Description: If user has selected both a label and data column then show next button.
	# 	Params: Label column name and data column names.
	# 	Returns: Invisible classname for next button if both columns not selected and visible class name if both are.
	# '''

	col1 = str(column1)
	col2 = str(column2)
	if col1 !='None' and col2 != 'None':
		return 'next-btn-visible'
	else:
		return 'next-btn-invis'

@app.callback(Output('clean-preview', 'children'),
	[Input('default-options', 'value'), 
	Input('add-options', 'value'),
	Input('stem-lem-options', 'value')]
	)

def clean(default, additional, stem_lem):
	# '''
	# 	Description: When user selects one of the additional cleaning options, iupdate the page with a live preview of the first entry after cleaning with selected options.
	# 	Params: list of selected default options, list of additional options, string of value: 'stem' or 'lemma'
	# 	Returns: Live update of first entry from dataframe after being cleaned
	# '''

	global interface_obj
	## Recreate instance of clean dataframe from non-clean to refresh with newly
	## selected options
	## = only first two columns
	interface_obj.clean_df = interface_obj.df.copy()
	
	## Do additional and default cleaning
	if additional:
		interface_obj.additional_clean(additional)
	## If stem or lem selected do those
	if stem_lem:
		interface_obj.stem_lem_clean(stem_lem)

	return (
		html.P('CLEANING PREVIEW'), 
		update_table(interface_obj.clean_df.head(1), interface_obj.clean_df.columns))

@app.callback(Output('graph-with-slider', 'figure'), 
	[Input('percent-slider', 'value')])

def update_graph(threshold):
	# '''
	# 	Description: Updates the threshold graph with only the words that are between the two user selected thersholds from the dash slider.
	# 	Params: Threshold tuple (lower threshold, upper threshold)
	# 	Returns: Graph data for only the words and word frequencies of the words between the two thresholds.
	# '''

	global interface_obj
	## Get bottom & top thresholds and rebuild the word/doc. freq. dataframe
	bottom_threshold = float(threshold[0]) * float(5) / float(100) * float(len(interface_obj.clean_df))
	top_threshold = float(threshold[1]) * float(5) / float(100) * float(len(interface_obj.clean_df))
	new_df1 = interface_obj.freq_matrix.where(interface_obj.freq_matrix[1] >= bottom_threshold).dropna()
	new_df2 = interface_obj.freq_matrix.where(interface_obj.freq_matrix[1] <= top_threshold).dropna()

	## Merge on words above bottom thresh. and words below top thresh.
	interface_obj.freq_matrix_keep = pd.merge(new_df1, new_df2, how='inner', on=[0])

	return {
		## Set x as doc. freq. of words within bottom and top thresholds
		'data' : [graph.Scattergl(
		x = interface_obj.freq_matrix_keep['1_y'],
		y = interface_obj.freq_matrix_keep.index,
		mode = 'markers',
		text = interface_obj.freq_matrix_keep[0]
		)],
		## Rebuild range of x and y axis with min and max of new dataframe
		'layout' : graph.Layout(
		xaxis = {'title' : 'Document Frequency'},
		yaxis = {'title' : 'Word Ids (ordered by doc. freq.)'},
		hovermode = 'closest',
	)
}

@app.callback(Output('bottom-thresh', 'children'), 
	[Input('percent-slider', 'value')])

def populate_bottom(value):
	# '''
	# 	Description: Populate the bottom list of excluded words that are outside of the threshold, when the slider is changed.
	# 	Params: Threshold tuple from slider value (lower threshold, upper threshold)
	# 	Returns: List of words that are below the user selected threshold.
	# '''

	global interface_obj
	threshold = float(value[0]) * float(5) / float(100) * float(len(interface_obj.clean_df))
	bottom_words = interface_obj.freq_matrix.where(interface_obj.freq_matrix[1] < threshold).dropna()
	bottom_words = bottom_words[0].tolist()[::-1]
	return ", ".join(word for word in bottom_words)

@app.callback(Output('top-thresh', 'children'), [Input('percent-slider', 'value')])

def populate_top(value):
	# '''
	# 	Description: Populate the top list of excluded words that are outside of the threshold, when the slider is changed.
	# 	Params: Threshold tuple from slider value (lower threshold, upper threshold)
	# 	Returns: List of words that are above the user selected threshold.
	# '''

	global interface_obj
	threshold = float(value[1]) * float(5) / float(100) * float(len(interface_obj.clean_df))
	top_words = interface_obj.freq_matrix.where(interface_obj.freq_matrix[1] > threshold).dropna()
	top_words = top_words[0].tolist()

	return ", ".join(word for word in top_words)

@app.callback(Output('cutoff-confirmation', 'children'),
				events=[Event('cutoff-btn', 'click')])

def cutoff_words():
	# '''
	# 	Description: Based on where the current slider values are, builds a list of words that are within the thresholds.
	# 				 Then remove all of the words that are not contained in this list of words from the uploaded data.
	# 	Params: None
	# 	Returns: Confirmation that the words have successfully been cutoff (removed) from the dataframe.
	# '''

	global interface_obj
	interface_obj.remove_cutoff()

	return html.P('Cut off completed successfully.')

@app.callback(Output('download-confirmation', 'children'),
			events = [Event('download-btn', 'click')]
	)

def download():
	# '''
	# 	Description: After user cleans (or doesn't) their data, when the download button is pressed, write clean dataframe out to disk.
	# 	Params: None
	# 	Returns: Confirmation that the data has been downloaded successfully.
	# '''

	global interface_obj
	file_name = 'cleaned_data'
	x = 1
	while file_name +'.csv' in files:
		file_name += str(x)
	file_name += '.csv'
	interface_obj.clean_df.to_csv(file_name)

	return html.P('Download completed successfully!')

@app.callback(Output('build-confirmation', 'children'),
				events=[Event('build-btn', 'click')],
				state = [State('model-choices', 'values')]
	)

def build_models(values):
	# '''
	# 	Description: Based on what models the user has selected to build, build the models inside of the backend object.
	# 	Params: Selected model values
	# 	Returns: Confirmation that the models have been built
	# '''

	tf_idf, lsa, lda, spacy = False, False, False, False
	if u'tf_idf' in values:
		tf_idf = True
	if u'lsa' in values:
		lsa = True
	if u'lda' in values:
		lda = True
	if u'spacy' in values:
		spacy = True

	global interface_obj
	lsa_nt = utils.get_num_of_topics(interface_obj.clean_df)
	if lsa_nt == 0:
		lsa_nt = 300
	lda_nt = lsa_nt / 3

	interface_obj.build_models(tf_idf, lsa, lsa_nt, lda, lda_nt, spacy)

	return html.P('Models built successfully!')

@app.callback(Output('similar-courses', 'children'),
				state = [State('amount-entry', 'value'), State('course-selection', 'value')],
				events = [Event('get-similar-btn', 'click')]        
)

def get_similar(amount, course_name):
	# '''
	# 	Description: Given a selected label (course) output the similar X amount of courses (where X is the amount the user gives).
	# 	Params: Amount of similar courses to get and the label name to find similar for.
	# 	Returns: List of amount of similar courses.
	# '''

	global interface_obj
	courses = interface_obj.iqss_model.get_similar(course_name, int(amount))
	## in form [(course1, score), (course2, score), ...]
	course_names = [course[0] for course in courses]
	
	return [html.Li(course) for course in course_names]

@app.callback(Output('analyze-content', 'children'),
			[Input('analyze-dropdown', 'value')]
	)

def show_analyzing_content(value):
	# '''
	# 	Description: Using the option that the user selects, show the content for that analyzing page.
	# 	Params: Value from dropdown of analyzing page options.
	# 	Returns: Content for the analyzing page.
	# '''

	global interface_obj
	if value == 'ents':
		interface_obj.load_ents()
		x = [label for label, texts in interface_obj.all_entities.items()]
		y = [len(texts) for label, texts in interface_obj.all_entities.items()]
		return html.Div([
			html.Div([
				dcc.Graph(
					id='entity-pie', 
					animate=True,
					figure = {
						'data' : [graph.Pie(
							labels = x,
							values = y,
							textinfo = 'percent'
						)]
						}
					),
				],
				className = 'entity-containers'),
			html.Div([
				dcc.Dropdown(
					options = [{'label' : label, 'value' : label} for label, texts in interface_obj.all_entities.items()],
					placeholder='Select entity label...',
					id = 'entity-choice'
				),
				html.Div(id='entity-table-container')],
				className = 'entity-containers')

			])

	elif value == 'topics':
		global interface_obj
		global cluster_obj
		cluster_obj = lda_cluster(interface_obj.clean_df)
		cluster_obj.make_lda(10, 1)
		return html.Div([
			html.Div([
				html.Div([
					html.P('# OF TOPICS'),
					dcc.Input(
						placeholder='# of topics',
						type='number',
						value = 10,
						id='vis-topics-input'
						)
					], className='half-container'),
				html.Div([
					html.P('# OF ITERATIONS'),
					dcc.Input(
						placeholder='# of iterations',
						type='number',
						value = 1,
						id='vis-iteration-input'
						)
					], className='half-container'),
				html.Div([
					html.Button('BUILD', id='build-vis-btn')
					], style={'clear' : 'both'})
				], id='lda-vis-top-container'),

			html.Div([
				html.Div([
					dcc.RadioItems(
						options=[{'label' : '2D', 'value' : '2d'},
								{'label' : '3D' , 'value' : '3d'}
						],
						value = '2d',
						id='2d3d'
						),
					dcc.Slider(
						min=0,
						max=20,
						step=.2,
						marks=[int(x) for x in range(0, 101, 5)],
						id = 'lda-vis-slider',
						value = 0,
						)
					],id = 'vis-options-container'),
				html.Div([
					dcc.Graph(id='lda-vis-graph', style={'marginTop' : '3%'})
					], className = 'three-fourth-container'),
				html.Div([
					html.P('TOPICS'),
					html.Ul(
							children = [], 
							id='lda-vis-legend', 
							style={'listStyleType' : 'none'}
						)
					], id='legend-container', className = 'fourth-container'),
				])
			])

@app.callback(Output('entity-table-container', 'children'),
			[Input('entity-choice', 'value')]
	)

def load_entity_table(label):
	# '''
	# 	Description: Allows the user to analyze the individual words within each category of entity.
	# 	Params: Selected category of entity from dropdown.
	# 	Returns: HTML table of entities within the entity category.
	# '''

	global interface_obj
	df = utils.get_dataframe(label, interface_obj.entity_dic)
	columns = df.columns
	return html.Table(
				[html.Tr([html.Th(label)])] +

				[html.Tr([html.Th(col) for col in columns])] +

				[html.Tr([
					html.Td(df.iloc[i][col]) for col in columns
					]) for i in range(len(df))], id='entity-table'
				)

@app.callback(Output('lda-vis-slider', 'value'),
	state = [State('vis-topics-input', 'value'), State('vis-iteration-input', 'value')],
	events = [Event('build-vis-btn', 'click')]
	)

def build_lda_vis(num_of_topics, iterations):
	# '''
	# 	Description: Builds the LDA model based on the user selected number of topics and iterations.
	# 	Params: Number of topics and number of iterations.
	# 	Returns: Sets the slider value (threshold) to 0.
	# '''

	global cluster_obj
	cluster_obj.make_lda(int(num_of_topics), int(iterations))
	return 0

@app.callback(Output('lda-vis-graph', 'figure'),
			[Input('lda-vis-slider', 'value'), Input('2d3d', 'value')]
	)	

def build_cluster_graph(slider_val, dim):
	# '''
	# 	Description: Builds the graph data for the LDA cluster visualizations.
	# 	Params: Threshold slider value and the dimension of 2d or 3d.
	# 	Returns: Figure and data for the LDA cluster graph visual.
	# '''

	global cluster_obj
	slider_val2 = float(slider_val) * float(5) / float(100)
	fig = cluster_obj.set_graph(slider_val2, dim)
	return fig

@app.callback(Output('lda-vis-legend', 'children'),
			[Input('lda-vis-slider', 'value'), Input('2d3d', 'value')]
	)

def build_legned(slider_val, dim):
	# '''
	# 	Description: Creates the legend that displays the words that correspond to each topic.
	# 				 Color coded to match the cluster color in the graph.
	# 	Params: Threshold slider value and the dimension of 2d or 3d.
	# 	Returns: List of words that correspond to each cluster.
	# '''

	global cluster_obj
	leg = cluster_obj.make_legend()

	list_of_lis = []
	for item, value in leg.items():
		list_of_lis.append(html.Li(item, style={'color' : value}))

	return list_of_lis

if __name__ == '__main__':
	app.run_server(debug=True)