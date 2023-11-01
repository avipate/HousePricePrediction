# Importing required libraries
import pandas as pd
import numpy as np
import joblib

# load the unseen / new dataset
data = pd.read_csv("Data/new_data.csv")

# Visualize the data
print(data.head())

# Drop the id variable
data.drop("Id", axis=1, inplace=True)

# Missing values
# categorical Variables
data["MSSubClass"] = data["MSSubClass"].astype("O")

# List of different groups of categorical variables
with_string_missing = ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]

with_frequent_category = {
    "MasVnrTpe": "None",
    "BsmtQual": "TA",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "Unf",
    "BsmtFinType2": "Unf",
    "Electrical": "SBrkr",
    "GarageType": "Attchd",
    "GarageFinish": "Unf",
    "GarageQual": "TA",
    "GarageCond": "TA",
}

# Replace missing values with new label: "Missing"
data[with_string_missing] = data[with_string_missing].fillna("Missing")

# replace missing values with the most frequent category
for var in with_frequent_category.keys():
    data[var].fillna(with_frequent_category[var], inplace=True)

# Numerical Variables
vars_with_na = {
    "LotFrontage": 69.8749374030,
    "MasVnrArea": 103.7974006128,
    "GarageYrBlt": 1978.29373674,
}

# replace missing values as we described above
for var in vars_with_na.keys():
    # add binary missing indicator (in train and test)
    data[var + '_na'] = np.where(data[var].isnull(), 1, 0)

    # replace missing values by the mean (in train and test)
    data[var].fillna(vars_with_na[var], inplace=True)

data[var]