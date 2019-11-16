import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Tokenizes and cleanes the input text message"""

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


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("messages", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extracting data needed for visuals
    category_counts = df.iloc[:,4:].sum().sort_values(ascending=False)
    category_names = list(df.iloc[:,4:].sum().sort_values(ascending=False).index)

    genre_direct = df[df["genre"]=="direct"].iloc[:,4:].sum()
    genre_news = df[df["genre"]=="news"].iloc[:,4:].sum()
    genre_social = df[df["genre"]=="social"].iloc[:,4:].sum()
    genre_df = pd.DataFrame([genre_direct, genre_news, genre_social],
                            index=["direct", "news", "social"])

    def generate_traces(df):
        """Generates traces for a plotly stacked bar chart"""

        traces = []
        for i in range(len(df.columns)):
            trace = Bar(
                    x = list(df.index),
                    y = df.iloc[:,i],
                    name = df.columns[i]
                    )
            traces.append(trace)

        return(traces)


    # creating visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Messages per Category',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    "tickangle":45,
                    "automargin":True
                }
            }
        },
        {
            'data': generate_traces(genre_df),
            'layout': {
                'barmode': 'stack',
                'title': 'Messages per Genre',
                'yaxis': {
                    'title': "Number of Messages"
                }
            }
        }
    ]

    # encoding plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
