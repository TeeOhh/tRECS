RECOMMENDATION SYSTEM DEVELOPER

Authors: Taylor Olson, Janie Neal, Christiana Prater-Lee, Eshita Nandini

This recommendation system developer guides the user through cleaning their data, building models, and ultimately creates a recommendation system (housed within the interface). The user can also visualize some the models and other features of their data.

INSTALLATION

The user should be using python 2.7+. Most packages required for the interface can be installed in the command line by navigating to the directory containing the reqs.txt file and running "pip install -r reqs.txt". Some packages require further installation. Links to their installation documentation are provided below.

Spacy: https://spacy.io/docs/usage/ 
Follow the instructions for downloading the english model.
 
NLTK:     http://www.nltk.org/data.html#
Follow the instructions to open the interactive installer and install the following corpora: wordnet, wordnet_ic, words, and stopwords.    


HOW TO START THE DASH INTERFACE

In order to start the interface, the user should navigate to the "interface" folder in the command line and call "python interface.py". The command window will provide a link that the user can then copy into the address bar of a browser of their choice. The interface should open in that window, and the user can begin using the features.


FEATURES OF THE INTERFACE AND HOW TO USE THEM

Upload your own data:
User can copy custom data (see sample data for desired format) into the interface folder or use some of the sets included in the interface folder.

Inside the interface the user should select the name of the file and press upload, then select the name of column within the file that contains the names of the objects that are described in the description column(selected next). The description column  should contain the text that will be analyzed. Click next step.

Choose your own cleaning options:
User is required to remove non-ascii characters and punctuation. There are also other optional cleaning features.
stemming vs. lemmatizing: https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html

Word cut off:
Numbers on the slider indicate the percentage of documents that the words appear in. Use the slider to remove words that occur too frequently or not frequently enough. 

Choose and Build your models:
Options are TF-IDF, LSA, LDA, and spaCy's built in similarity function.

Recommendation: 
Uses the models built in the previous step to provide the selected number of most similar objects to the selected object.

Visualizations:
The following visualizations have been connected to the interface: entity visualization, lda clustering. Please see their descriptions in the description of visualizations.py.


CONTENTS OF MASTER FOLDER

Interface folder:

Multiple sample datasets.

Interface.py: contains the code needed for front end of application, including: HTML layout, Dash components and event handlers, calls to backend. Contains iqss_interface object (below).

Iqss_interface.py: contains all coded needed for back end of application, including: user loaded data frame, cleaned data frame, and built models. Contains two objects, the iqss_interface object (holds the data frames) and the model_object (holds the necessary df, tf-idf, lda, and spacy models). 

Utils.py: contains all methods used for manipulating data frames, including: converting to term and document frequency matrix, cleaning the data frame, extracting entities from the data frame, and determining the number of topics based on the contents of the data frame.
    
Visualizations.py: contains all code necessary to build the various visualizations, including: 
the LDA cluster graph: http://brandonrose.org/clustering  
the similarity graph: a graph that plots the similarity score of an object on the x axis, and the group which the object is part of is plotted on the y axis
LDA topic distribution: https://pyldavis.readthedocs.io/en/latest/readme.html#installation

