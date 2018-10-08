# Disaster Response Pipeline Project
# By Rakan Alkheliwi
### Link:
https://github.com/ra2309/DisasterResponse
### Objective:
The project analyzes collection of tweets and determine what kind of disaster is involved.
### Files:
The project contains three folders:
- app
contains Flask server and plotly to plot the graphs
- data
contains data and process_data.py which is an ETL pipleine in addition to resulting database
- models
contains train_classifier.py which is an ML pipeline and resulting pkl file from model
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
