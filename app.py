from flask import Flask, request, jsonify, render_template
import joblib
from pyforest import *
import numpy as np
from main import manipulateFeatureNames
import requests

#naming our app
HomeHealer= Flask(__name__)

#loading the pickle file for creating the web app
model= joblib.load(open("trained_dt_model.pkl", "rb"))

# defining the different pages of html and specifying the features required to be filled in the html form
@HomeHealer.route("/")
def home():
    return render_template("index.html")


def oneHotConversion(values):
    # load in the csv file from 
    feature_framework = np.loadtxt("one_hot_framework.csv", dtype=str, usecols=0)
    # reduce discrepancies in inputted feature names
    values = manipulateFeatureNames(values)
    #get feature locations within the framework
    locs= [np.where((val) == np.asarray(feature_framework))[0] for val in values]
    # initialise feature vector
    feature_vec = np.zeros(len(feature_framework))
    # update values accordingly
    feature_vec[locs] = 1
    return feature_vec


# defining a prediction function
@HomeHealer.route("/predict", methods=["POST"])
def predict():
    '''
    This function acts is the core flask function that interacts with the front end. Values
    are received from the user and inputed in the train model. The output is then 
    communicated.
    '''
    # extract inputed features
    rating= request.form.get('rating')
    housetype= request.form.get('housetype')
    postcode = request.form.get('postcode')

    # create a list of inputed features
    inputed_features = [rating,postcode,housetype]

    # Check if values entered for every input feature
    if None in inputed_features or '' in inputed_features:

        # Return erro prompt if not complete
        return render_template("index.html", prediction_text= 'Please select a value for all inputs.')
    else:

        # Making a get request to get Local Authority from inputed postcode
        response = requests.get('https://findthatpostcode.uk/postcodes/{}.json'.format(str(postcode))).json()
 
        # Obtain local Authority through directory search
        local = str(response['data']['attributes']['laua_name'])

        # create a list of categorical features
        string_features = [rating,local,housetype]

        # convert categorical features to numerical using a previously saved one-hot encoding framework
        final_features= [np.array(oneHotConversion(string_features))]
        
        # obtain prediction for the inputed features
        prediction= model.predict(final_features)

        # return these predictions to the user
        output= round(prediction[0], 2)
        return render_template("index.html", prediction_text= "Your property could potentially save up to {} KW/h of energy per year.".format(output))

if __name__== "__main__":
    HomeHealer.run(debug=True)