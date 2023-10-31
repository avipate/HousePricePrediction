# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats


class HousePricePrediction:
    def __init__(self, path="Data/HousePrice.csv"):
        self.df = pd.read_csv(path)
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    # Splitting the dataset
    def feature_engineering(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.df.drop(["Id", "SalePrice"], axis=1),
            self.df["SalePrice"],
            test_size=0.1,
            random_state=0,
        )
        # Capture teh categorical columns
        cat_vars = [var for var in self.df.columns if self.df[var].dtype == "O"]
        # let's add MSSSubClass to the list of categorical variables
        cat_vars = cat_vars + ["MSSubClass"]

        # Cast all variables as categorical
        self.x_train[cat_vars] = self.x_train[cat_vars].astype("O")
        self.x_test[cat_vars] = self.x_test[cat_vars].astype("O")

        # make a list of the categorical variables that contain missing values
        cat_vars_with_na = [
            var for var in cat_vars if self.x_train[var].isnull().sum() > 0
        ]
        print(
            self.x_train[cat_vars_with_na].isnull().mean().sort_values(ascending=False)
        )

        # variables to impute with the string missing
        with_string_missing = [
            var for var in cat_vars_with_na if self.x_train[var].isnull().mean() > 0.1
        ]

        # variables to impute with most frequent category
        with_frequent_category = [
            var for var in cat_vars_with_na if self.x_train[var].isnull().mean() < 0.1
        ]

        # replace missing values with new labels: "Missing"
        self.x_train[with_string_missing] = self.x_train[with_string_missing].fillna(
            "Missing"
        )
        self.x_test[with_string_missing] = self.x_test[with_string_missing].fillna(
            "Missing"
        )

        for var in with_frequent_category:
            # there can be more than 1 mode in a variable
            # we take first one with [0]
            mode = self.x_train[var].mode()[0]

            self.x_train[var].fillna(mode, inplace=True)
            self.x_test[var].fillna(mode, inplace=True)

        # Check that we have no missing information in the engineered variables
        print(self.x_train[cat_vars_with_na].isnull().sum())

        # Now numerical variables
        num_vars = [
            var
            for var in self.x_train.columns
            if var not in cat_vars and var != "SalePrice"
        ]

        print(len(num_vars))

        # make a list with the numerical variables that contain missing values
        var_with_na = [var for var in num_vars if self.x_train[var].isnull().sum() > 0]

        # Print percentage of missing values per variable
        print(self.x_train[var_with_na].isnull().mean())

        # Replace missing values
        for var in var_with_na:
            # calculate the mean using the train set
            mean_val = self.x_train[var].mean()
            print(var, mean_val)

            # add binary missing indicator (in train and test)
            self.x_train[var + '_na'] = np.where(self.x_train[var].isnull(), 1, 0)
            self.x_test[var + '_na'] = np.where(self.x_test[var].isnull(), 1, 0)

            # replace missing values by the mean (in train and test)
            self.x_train[var].fillna(mean_val, inplace=True)
            self.x_test[var].fillna(mean_val, inplace=True)

        # check that we have no more missing values in the engineered variables
        print(self.x_train[var_with_na].isnull().sum())
        print(self.x_test[var_with_na].isnull().sum())

        print(self.x_train[['LotFrontage_na', 'MasVnrArea_na', 'GarageYrBlt_na']].head())

        # Part 2
        # Capture elapsed time
        def elapsed_year(df, var):
            # capture difference between the year variable
            # and the year in which the house was sold
            df[var] = df['YrSold'] - df[var]
            return df

        for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
            self.x_train = elapsed_year(self.x_train, var)
            self.x_test = elapsed_year(self.x_test, var)

        # now we drop YrSold
        self.x_train.drop(['YrSold'], axis=1, inplace=True)
        self.x_test.drop(['YrSold'], axis=1, inplace=True)

        # Numerical variable transformation
        # Logarithmic transformation
        for var in ["LotFrontage", "1stFlrSF", "GrLivArea"]:
            self.x_train[var] = np.log(self.x_train[var])
            self.x_test[var] = np.log(self.x_test[var])

        # Check that test set does not contain null values in the engineered variables
        null = [var for var in ["LotFrontage", "1stFlrSF", "GrLivArea"] if self.x_test[var].isnull().sum()]
        print(null)

        null = [var for var in ["LotFrontage", "1stFlrSF", "GrLivArea"] if self.x_train[var].isnull().sum()]
        print(null)


if __name__ == "__main__":
    model = HousePricePrediction()
    model.feature_engineering()


user_input = input("Enter your dish name: ")
print(user_input)
