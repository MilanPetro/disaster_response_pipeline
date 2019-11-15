## Disaster Response Pipeline Project
### Building ETL and Machine Learning pipelines to classify text messages during disasters

The goal of the following project was to use real text messages that were sent during disaster events to create a machine learning model that would be able to classify messages into different categories. This could then help disaster response professionals to filter trough the myriad of messages that are being sent during these situations, to find and focus on the messages with the highest priority for them.

The data set used in this project contains pre-labeled real messages that were sent during disasters and were obtained from the data annotation platform [Figure Eight](https://www.figure-eight.com/). The messages were either send via social media or directly to disaster response organization.

I built an ETL pipeline that loads and processes the data and saves it in a SQL database. A subsequent machine learning pipeline then reads from this database to create and save a supervised machine learning model. In the last step a web app is being generated that uses this model, so that an emergency worker can input a new message to be classified for 36 different categories.

The Project has the following components:

1. ETL Pipeline

    The process_data.py file in the data/ folder contains the Python script for the data cleaning pipeline:

    - Loads the messages and categories datasets from the data folder
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database


2. ML Pipeline

    The Python script train_classifier.py in the model/ folder builds the machine learning pipeline:

    - Loads data from the SQLite database
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a random forest classifier using grid search
    - Outputs the results on the test set
    - Exports the final model as a pickle file


3. Flask Web App

    The run.py file in the app/ folder generates the web app. It allows one to enter text messages that will be, if possible, ascribed to one of the 36 categories the model was trained with.

### Instructions:

An Anaconda distribution with Python 3 should be sufficient to run this code.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL pipeline that cleans the data and stores in it in a SQL database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains the classifier and saves it:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
