import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
import joblib
import string
import pymongo

### Functions ###

def manipulateFeatureNames(names):
    ''''
    This function manipulates user inputs to map to the features in the framework.
    '''
    new_names = [ n.translate(str.maketrans('', '', string.punctuation)).replace(" ", "").lower() for n in names]
    return new_names

### Code ###

print('Extracting data from MongoDB...')

# Load in dataset from mongodb
client = pymongo.MongoClient("mongodb+srv://ai_team:homehealer@aianda.qz6bpjf.mongodb.net/test", connect=False)
   
# Database name
db = client["AIandA"]
   
# Collection name
mongo_collection = db["homehealer"]
print(list(mongo_collection.find()))
# Convert to dataframe
housing_df = pd.DataFrame(list(mongo_collection.find()))

print('Formatting database...')

# Create a saved energy column
housing_df['saved-energy-potential'] = np.abs(housing_df['energy-consumption-potential'].to_numpy(dtype=int) - housing_df['energy-consumption-current'].to_numpy(dtype=int))

# Choose parameters
datapoints_df = housing_df[["current-energy-rating", 
                        "built-form", 
                        "local-authority-label"]]
# Set target values
labels_df = housing_df[["saved-energy-potential"]]


# Creating a one hot encoding for all categorical variables
oh1 = pd.get_dummies(datapoints_df['current-energy-rating'])
oh2 = pd.get_dummies(datapoints_df['built-form'])
oh3 = pd.get_dummies(datapoints_df['local-authority-label'])

# merge the dataframes
one_hot_df = pd.concat([pd.concat([oh2,oh3], axis=1),oh1], axis=1)
np.savetxt(r"one_hot_framework.csv", manipulateFeatureNames(list(one_hot_df.columns)), fmt='%s', delimiter=",")

print('Training saved energy model...')

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(one_hot_df, labels_df, test_size=0.33, random_state=42)

# Train regression model
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)

# pickling the model
joblib.dump(clf, "trained_dt_model.pkl")


### Working on a seperate model to predict carbon emission potential ###

# Create a saved carbon column
housing_df['saved-carbon-potential'] = np.abs(housing_df['carbon-emission-potential'].to_numpy(dtype=int) - housing_df['carbon-emission-current'].to_numpy(dtype=int))


# labels
carbon_labels_df = housing_df[["saved-carbon-potential"]]

print('Training saved carbon model...')

# Split the dataset
X_train_carbon, X_test_carbon, y_train_carbon, y_test_carbon = train_test_split(one_hot_df, carbon_labels_df, test_size=0.33, random_state=42)

# Train regression model
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train_carbon, y_train_carbon)

# pickling the model
joblib.dump(clf, "trained_carbon_dt_model.pkl")
