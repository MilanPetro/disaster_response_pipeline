# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import numpy as np
import pandas as pd
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sqlalchemy import create_engine


def load_data(database_filepath):
    """ Loading dataset from SQL database.
    Input: filepath to sqlite database
    Output: feature variables X, target variables Y , category names"""

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df["message"].values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns.values
    return X, Y, category_names


def tokenize(text):
    """Tokenizes and cleanes the text message data
    Input: text messages X
    Output: cleaned tokens"""

    # removing punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]" ," ", text)

    # tokenizing
    tokens = word_tokenize(text)

    # lemmatizing and cleaning tokens (lower case and without spaces)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    # removing stop words
    clean_tokens = [t for t in clean_tokens if t not in
                    stopwords.words("english")]

    return clean_tokens


def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())])

    parameters = {"clf__n_estimators": [100, 200, 400],
                  "clf__min_samples_split": [2, 4]}

    classifier = GridSearchCV(pipeline, cv=3, param_grid=parameters, verbose=8)

    return classifier


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model. Prints out the f1 score, precision and recall
    for the test set for each category."""

    Y_pred = model.predict(X_test)

    for i in range(len(Y_test[0])):
        print(list(category_names)[i])
        print(classification_report(Y_test[:,i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """Saves the model as a pickle file"""

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
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
