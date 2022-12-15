#this is an old script you can plug back into app.py to get check.html to run
#it was a test page i made to test out a possible check answers page
#dosent work because i dont know how to get information from a form across two pages

# @app.route("/check", methods = ['GET', 'POST'])

# def check():
#     energy= request.form.get('energy')
#     housetype= request.form.get('house-type')
#     postcode = request.form.get('postcode')
    
#     inputed_features = [energy,postcode,housetype]

#     if None in inputed_features or '' in inputed_features:

#         # Return erro prompt if not complete
#         return render_template("test.html", prediction_text= 'Please select a value for all inputs.')
#     else:

#         # Making a get request to get Local Authority from inputed postcode
#         response = requests.get('https://findthatpostcode.uk/postcodes/{}.json'.format(str(postcode))).json()
 
#         # Obtain local Authority through directory search
#         local = str(response['data']['attributes']['laua_name'])

#         # Obtain logitude and latitude of the postcode
#         lon = str(response['data']['attributes']['location']['lon'])
#         lat = str(response['data']['attributes']['location']['lat'])

#     return render_template("check.html", home_type = request.form['house-type'], energy_rating = request.form['energy'], postcode = request.form['postcode'], lon = lon, lat = lat )
