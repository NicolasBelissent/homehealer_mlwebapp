from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from model import manipulateFeatureNames
import requests

#naming our app
app= Flask(__name__)

#loading the pickle file for creating the web app
model= joblib.load(open("trained_dt_model.pkl", "rb"))

# defining the different pages of html and specifying the features required to be filled in the html form
@app.route("/")
def home():
    # if request.method == "POST":
    #     home_type=request.form.get("house-type")
    #     energy_rating = request.form.get("energy")
    #     postcode = request.form.get("postcode")
    #     return "your house is " + home_type + " with energy rating " + energy_rating+ " in " + postcode
    return render_template("start.html")


@app.route('/about', methods = ['GET', "POST"])
def about(): 
    return render_template('about.html')


@app.route("/results", methods = ['GET', 'POST'])
def results():
    # extract inputed features
    rating= request.form.get('energy')
    housetype= request.form.get('house-type')
    postcode = request.form.get('postcode')

    # create a list of inputed features
    inputed_features = [rating,postcode,housetype]

    # Check if values entered for every input feature
    if None in inputed_features or '' in inputed_features:

        # Return erro prompt if not complete
        return render_template("error.html", error_message= 'Please select a value for all inputs.')
    else:

        # Making a get request to get Local Authority from inputed postcode
        response = requests.get('https://findthatpostcode.uk/postcodes/{}.json'.format(str(postcode))).json()
 
        try:

            # Obtain local Authority through directory search
            local = str(response['data']['attributes']['laua_name'])

            # Obtain clean postcode
            clean_postcode = str(response['data']['attributes']['pcds'])

        except KeyError:
            return render_template("error.html", error_message= 'Please enter a valid postcode.')

         # Obtain logitude and latitude of the postcode
        lon = str(response['data']['attributes']['location']['lon'])
        lat = str(response['data']['attributes']['location']['lat'])

        # create a list of categorical features
        string_features = [rating,local,housetype]
        # convert categorical features to numerical using a previously saved one-hot encoding framework
        final_features= [np.array(oneHotConversion(string_features))]
        
        # obtain prediction for the inputed features
        prediction= model.predict(final_features)

        # return these predictions to the user
        output= round(prediction[0], 2)
        
        return render_template("results.html", home_type = request.form['house-type'], 
        energy_rating = request.form['energy'], postcode = clean_postcode,
        prediction_text= "Your property could potentially save up to {} KW/h of energy per year.".format(output),
        lon = lon, lat = lat)


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


if __name__== "__main__":
    app.run(debug=True)
