import sys
# To load data from sql db 
from sqlalchemy import create_engine

#To work with dataframes
import pandas as pd

# To tokenize text
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#To create a pipeline
from sklearn.pipeline import Pipeline

# Data transformers
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Multi output classifier using random forest TODO:how it works
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

#Split data into training and test sets
from sklearn.model_selection import train_test_split

#Metrics
from sklearn.metrics import classification_report

#Save model
import joblib


def load_data(database_filepath):
	"""
	Load dataset from file
	Input :
		- database_filepath
	Output :
		- X : features of dataset
		- Y : targets of dataset
		- category_names
	"""
	engine = create_engine('sqlite:///{}.db'.format(database_filepath))
	con=engine.connect()
	df = pd.read_sql_table(database_filepath,con=con)
	X = df['message'] 
	Y = df.drop(['message','id','original','genre'],axis=1)
	category_names = Y.columns.tolist()
	return X,Y,category_names

def tokenize(text):
	"""
	Tokenize input text
	Input:
		- text : raw text
		- output : list of tokenized text
	"""
	lemmatizer=WordNetLemmatizer()
	return [ lemmatizer.lemmatize(lemmatizer.lemmatize(tok).lower().strip(),pos='v') for tok in word_tokenize(text)]

def build_model():
	"""
	Build a pipeline including data transform and data classification steps
	Transform stages used : 
		- CountVectorizer
		- TfidfTransformer
	Classifier stages used:
		- MultiOutputClassifier
			* This uses RandomForestClassifier
	Output :
		- pipeline
	"""
	pipeline = Pipeline([
	('vect',CountVectorizer(tokenizer=tokenize)),
	('tfidf',TfidfTransformer()),
	('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=4,verbose=10)))
	], verbose=True)
	return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
	"""
	Report performance of model
	Input:
		- model : the model (pipeline) we have used
		- X_test : test features
		- Y_test : test targets
		- category_names
	Output: NA
	"""
	y_pred=model.predict(X_test)
	print(classification_report(Y_test,y_pred,target_names=category_names))

def save_model(model, model_filepath):
	""" Save model to file 'model_filepath' """
	joblib.dump(model, model_filepath)
	#pickle.dump(model, open(model_filepath, 'wb'))


def main():
	if len(sys.argv) == 3:
		database_filepath, model_filepath = sys.argv[1:]
		print('Loading data...\n	DATABASE: {}'.format(database_filepath))
		X, Y, category_names = load_data(database_filepath)
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
		
		print('Building model...')
		model = build_model()
		
		print('Training model...')
		model.fit(X_train, Y_train)
		
		print('Evaluating model...')
		evaluate_model(model, X_test, Y_test, category_names)

		print('Saving model...\n	MODEL: {}'.format(model_filepath))
		save_model(model, model_filepath)

		print('Trained model saved!')

	else:
		print('Please provide the filepath of the disaster messages database '\
			  'as the first argument and the filepath of the pickle file to '\
			  'save the model to as the second argument. \n\nExample: python '\
			  'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
	main()