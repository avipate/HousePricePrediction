# Importing required libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score


class HousePricePrediction:
    def __init__(self, path="Data/HousePrice.csv"):
        self.df = pd.read_csv(path)
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    # Splitting the dataset
    def data_preprocessing(self):
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
            self.x_train[var + "_na"] = np.where(self.x_train[var].isnull(), 1, 0)
            self.x_test[var + "_na"] = np.where(self.x_test[var].isnull(), 1, 0)

            # replace missing values by the mean (in train and test)
            self.x_train[var].fillna(mean_val, inplace=True)
            self.x_test[var].fillna(mean_val, inplace=True)

        # check that we have no more missing values in the engineered variables
        print(self.x_train[var_with_na].isnull().sum())
        print(self.x_test[var_with_na].isnull().sum())

        print(
            self.x_train[["LotFrontage_na", "MasVnrArea_na", "GarageYrBlt_na"]].head()
        )

        # Part 2
        # Capture elapsed time
        def elapsed_year(df, var):
            # capture difference between the year variable
            # and the year in which the house was sold
            df[var] = df["YrSold"] - df[var]
            return df

        for var in ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]:
            self.x_train = elapsed_year(self.x_train, var)
            self.x_test = elapsed_year(self.x_test, var)

        # now we drop YrSold
        self.x_train.drop(["YrSold"], axis=1, inplace=True)
        self.x_test.drop(["YrSold"], axis=1, inplace=True)

        # Numerical variable transformation
        # Logarithmic transformation
        for var in ["LotFrontage", "1stFlrSF", "GrLivArea"]:
            self.x_train[var] = np.log(self.x_train[var])
            self.x_test[var] = np.log(self.x_test[var])

        # Check that test set does not contain null values in the engineered variables
        null = [
            var
            for var in ["LotFrontage", "1stFlrSF", "GrLivArea"]
            if self.x_test[var].isnull().sum()
        ]
        print(null)

        null = [
            var
            for var in ["LotFrontage", "1stFlrSF", "GrLivArea"]
            if self.x_train[var].isnull().sum()
        ]
        print(null)

        # Binarize skewed variables
        skewed = [
            "BsmtFinSF2",
            "LowQualFinSF",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "MiscVal",
        ]

        for var in skewed:
            # map the variable values into 0 to 1
            self.x_train[var] = np.where(self.x_train[var] == 0, 0, 1)
            self.x_test[var] = np.where(self.x_test[var] == 0, 0, 1)

        # Categorical variables
        # Apply mapping
        # re-map strings to numbers, which determine quality
        qual_mappings = {
            "Po": 1,
            "Fa": 2,
            "TA": 3,
            "Gd": 4,
            "Ex": 5,
            "Missing": 0,
            "NA": 0,
        }

        qual_vars = [
            "ExterQual",
            "ExterCond",
            "BsmtQual",
            "BsmtCond",
            "HeatingQC",
            "KitchenQual",
            "FireplaceQu",
            "GarageQual",
            "GarageCond",
        ]

        for var in qual_vars:
            self.x_train[var] = self.x_train[var].map(qual_mappings)
            self.x_test[var] = self.x_test[var].map(qual_mappings)

        exposure_mappings = {"No": 1, "Mn": 2, "Av": 3, "Gd": 4}
        var = "BsmtExposure"

        self.x_train[var] = self.x_train[var].map(exposure_mappings)
        self.x_test[var] = self.x_test[var].map(exposure_mappings)

        finish_mappings = {
            "Missing": 0,
            "NA": 0,
            "Unf": 1,
            "LwQ": 2,
            "Rec": 3,
            "BLQ": 4,
            "ALQ": 5,
            "GLQ": 6,
        }
        finish_var = ["BsmtFinType1", "BsmtFinType2"]

        for var in finish_var:
            self.x_train[var] = self.x_train[var].map(finish_mappings)
            self.x_test[var] = self.x_test[var].map(finish_mappings)

        garage_mappings = {"Missing": 0, "NA": 0, "Unf": 1, "RFn": 2, "Fin": 3}

        var = "GarageFinish"

        self.x_train[var] = self.x_train[var].map(garage_mappings)
        self.x_test[var] = self.x_test[var].map(garage_mappings)

        fence_mappings = {
            "Missing": 0,
            "NA": 0,
            "MnWw": 1,
            "GdWo": 2,
            "MnPrv": 3,
            "GdPrv": 4,
        }
        var = "Fence"

        self.x_train[var] = self.x_train[var].map(fence_mappings)
        self.x_test[var] = self.x_test[var].map(fence_mappings)

        null = [
            var for var in self.x_train.columns if self.x_train[var].isnull().sum() > 0
        ]
        print(null)

        # Remove Rare Labels
        # Capture all quality variables
        qual_vars = qual_vars + finish_var + ["BsmtExposure", "GarageFinish", "Fence"]

        # capture the remaining categorical variables
        # (those that we did not re-map)
        cat_others = [var for var in cat_vars if var not in qual_vars]

        print(len(cat_others))

        # Function that fins the labels that are shared by more than a certain % of the house in the dataset
        def find_frequent_labels(df, var, rare_perc):
            self.df = self.df.copy()

            tmp = self.df.groupby(var)[var].count() / len(self.df)

            return tmp[tmp > rare_perc].index

        for var in cat_others:
            # find the frequent categories
            frequent_ls = find_frequent_labels(self.x_train, var, 0.01)
            print(var, frequent_ls)

            # replace rare categories by the string "Rare"
            self.x_train[var] = np.where(
                self.x_train[var].isin(frequent_ls), self.x_train[var], "Rare"
            )
            self.x_test[var] = np.where(
                self.x_test[var].isin(frequent_ls), self.x_test[var], "Rare"
            )

        # Encoding of categorical variables
        def replace_categories(train, test, y_train, var, target):
            tmp = pd.concat([self.x_train, y_train], axis=1)

            # Order the categories in a variables from that with the lowest
            # house sale price, to that with the highest
            ordered_labels = tmp.groupby([var])[target].mean().sort_values().index

            # Create a dictionary of ordered categories to integer values
            ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}

            print(var, ordinal_label)

            # use the dictionary to replace the categorical strings by integers
            train[var] = train[var].map(ordinal_label)
            test[var] = test[var].map(ordinal_label)

        for var in cat_others:
            replace_categories(
                self.x_train, self.x_test, self.y_train, var, "SalePrice"
            )
        # Checking
        null = [
            var for var in self.x_train.columns if self.x_train[var].isnull().sum() > 0
        ]
        print(null)
        null = [
            var for var in self.x_test.columns if self.x_test[var].isnull().sum() > 0
        ]
        print(null)

    def feature_engineering(self):
        self.data_preprocessing()
        # create scaler
        scaler = MinMaxScaler()

        # fit the scaler to the train set
        scaler.fit(self.x_train)

        # Transform the train and test
        # sklearn return numpy arrays, so we wrap the array with a pandas dataframe
        self.x_train = pd.DataFrame(
            scaler.transform(self.x_train), columns=self.x_train.columns
        )
        self.x_test = pd.DataFrame(
            scaler.transform(self.x_test), columns=self.x_test.columns
        )
        print(self.x_train.head())

        # Let's now save the train and test sets to csv
        self.x_train.to_csv("Data/xtrain.csv", index=False)
        self.x_test.to_csv("Data/xtest.csv", index=False)

        self.y_train.to_csv("Data/ytrain.csv", index=False)
        self.y_test.to_csv("Data/ytest.csv", index=False)

        joblib.dump(scaler, "minmax_scaler.joblib")

    def ml_pipeline(self):
        x_train = pd.read_csv("Data/xtrain.csv")
        x_test = pd.read_csv("Data/xtest.csv")

        # Load the target
        y_train = pd.read_csv('Data/ytrain.csv')
        y_test = pd.read_csv('Data/ytest.csv')

        # Feature selection steps
        # First, we specify the Lasso Regression Model and select a suitable alpha (equivalent of penalty)
        # Then we use the selectFrom Model object from sklearn, which will select automatically the
        # features which coefficients are non-zero

        sel_ = SelectFromModel(Lasso(alpha=0.001, random_state=0))

        # train Lasso model and select features
        sel_.fit(x_train, y_train)
        print(sel_.get_support().sum())

        # Let's print the number of total and selected features
        selected_feats = x_train.columns[(sel_.get_support())]

        # let's print some stats
        print(f'Total features: {x_train.shape[1]}')
        print(f'Selected features: {len(selected_feats)}')
        print(f'Features with coefficients shrank to zero: {np.sum(sel_.estimator_.coef_ == 0)}')
        print(selected_feats)

        pd.Series(selected_feats).to_csv('Data/selected_features.csv', index=False)

        # Load the pre-selected features
        features = pd.read_csv('Data/selected_features.csv')
        features = features['0'].to_list()

        # model
        lin_model = Lasso(alpha=0.001, random_state=0)

        # Train the model
        lin_model.fit(x_train, y_train)

        # Prediction
        pred = lin_model.predict(x_train)

        print(f'Train mse: {float(mean_squared_error(y_train, pred))}')
        print(f'Train r2 score: {float(r2_score(y_train, pred))}')

        # Make prediction for test set
        pred = lin_model.predict(x_test)

        print(f'Test mse: {float(mean_squared_error(y_test, pred))}')
        print(f'Test r2 score: {float(r2_score(y_test, pred))}')

        # Save the model
        joblib.dump(lin_model, 'Models/linear_regression.joblib')


if __name__ == "__main__":
    model = HousePricePrediction()
    model.ml_pipeline()
