import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    Input: database file location
    Output: X, Y and column names
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(database_filepath,con=engine)
    X = df['message'].values
    Y = df.drop(['message','genre','original','id'],axis=1).values
    categories = df.drop(['message','genre','original','id'],axis=1).columns
    return X,Y,categories

def tokenize(text):
    """
    Input: text
    Output: tokenized text
    Procedure: the algorithm uses NLTK library to tokenize and clean text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = str(lemmatizer.lemmatize(tok).lower().strip())
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Output: GridSearch pipeline
    Procedure: the pipeline contains countvectorizer, TF-IDF transformer, and multioutput decission tree
               the parameters are different sets of values for countvectorizer and TF-IDF transformer
               GridSearchCV runs on both variables to find optimal parameters
    """
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfdif',TfidfTransformer()),
    ('clf',MultiOutputClassifier(DecisionTreeClassifier(max_depth=1),n_jobs=-1))
    ])
    parameters = {
    'vect__ngram_range':((1,1),(1,2)),
    'tfdif__use_idf':(True,False),
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input: model, X_test, Y_test, column names
    Output: printout of recall, prcession, and accuracy of model
    """
    y_preds = model.predict(X_test)
    for i in range(len(y_preds.T)):
        print(category_names[i])
        score = classification_report(Y_test.T[i],y_preds.T[i])
        print(score)


def save_model(model, model_filepath):
    """
    Input: model and model filepath
    Output: pkl file saved in model filepath
    """
    pickle.dump(model,open(model_filepath, 'wb'))


def main():
    """
    Run the procedures of loading, building model, evaluating, and dumping model
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()