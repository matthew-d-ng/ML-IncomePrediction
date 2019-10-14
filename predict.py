from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas

# CSV file paths
results_file = "results/tcd ml 2019-20 income prediction submission file.csv"
without_labels = "tcd ml 2019-20 income prediction test (without labels).csv"
with_labels = "tcd ml 2019-20 income prediction training (with labels).csv"

all_columns = [
    "Instance", "Year of Record", "Gender", "Age",
    "Country", "Size of City", "Profession", "University Degree",
    "Wears Glasses", "Hair Color", "Body Height [cm]"
]
no_str_columns = [
    "Instance", "Year of Record", "Age",
    "Size of City", "Wears Glasses", "Body Height [cm]"
]
ohe_columns = [
    "Country", "Profession", "Gender", "Hair Color"#, "University Degree"
]
input_columns = [
    "Year of Record", "Gender", "Age",
    "Country", "Size of City", "Profession", "University Degree",
    "Wears Glasses", "Hair Color", "Body Height [cm]"
]
target_columns = [
    "Income in EUR"
]

def lin_model(labelled_data, unlabelled_data):
    """ Parameters: training dataframe, unknown dataframe
        Returns: results dataframe

        Replaces NaN with ffill, one hot encodes non-numeric fields, 
        80/20 splits data to help verify model, uses LassoCV model
    """
    print("cleaning data...")
    clean_labelled = labelled_data.dropna()
    clean_unlabelled = unlabelled_data[all_columns]
    clean_unlabelled = clean_unlabelled.fillna(method="ffill")
    # clean_unlabelled = clean_unlabelled.fillna("None")

    # remove some columns
    # clean_labelled = drop_columns(clean_labelled)
    # clean_unlabelled = drop_columns(clean_unlabelled)

    print("one hot encoding data...")
    # One hot encoding
    ohe = OneHotEncoder(
        categories="auto", 
        handle_unknown="ignore",
        sparse=False
    )
    clean_labelled = encode_training(ohe, clean_labelled)
    clean_unlabelled = encode_testing(ohe, clean_unlabelled)

    unknown_data = clean_unlabelled.drop(["Instance"], axis=1)

    print("splitting data into train and test...")
    # 80/20 split
    split = split_data(clean_labelled)
    train_data, train_target, test_data, test_target = split

    print("scaling values...")
    # scaling values
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    unknown_data = scaler.transform(unknown_data)

    print("fitting model...")
    # fit model
    lasso = lm.LassoCV(cv=5)
    lasso.fit(train_data, train_target)

    print("analysing test results...")
    # validate test
    test_result = lasso.predict(test_data)
    error = np.sqrt(mean_squared_error(test_target, test_result))
    print("Root mean squared error of test data: ", error)

    print("predicting unknown data...")
    # predict and format
    values = lasso.predict(unknown_data)
    results = pandas.DataFrame({
        "Instance": clean_unlabelled["Instance"].values,
        "Income": values.flatten()
    })
    print("Finished.")
    return results

def encode_training(ohe, dataframe):
    """ Perform One Hot Encoding on labelled dataframe"""
    x = constrain_col_vals(dataframe)
    arr = ohe.fit_transform(x[ohe_columns])
    x = merge_encoding(ohe, x, arr)
    return x

def encode_testing(ohe, dataframe):
    """ Perform One Hot Encoding on unlabelled dataframe """
    x = constrain_col_vals(dataframe)
    arr = ohe.transform(x[ohe_columns])
    x = merge_encoding(ohe, x, arr)
    return x

def merge_encoding(ohe, dataframe, arr):
    """ Insert encoded values back into original dataframe """
    col_names = ohe.get_feature_names(ohe_columns)
    encoded = pandas.DataFrame(arr, columns = col_names)
    dataframe = dataframe.drop(ohe_columns, axis=1).reset_index(drop=True)
    dataframe = dataframe.join(encoded)
    return dataframe

def constrain_col_vals(dataframe):
    """ Standardise input variants of necessary categories """
    x = dataframe

    values = {"male": 0, "female": 1, "other": 2}
    x.loc[:, "Gender"] = x["Gender"].map(values)
    x.loc[:, "Gender"] = x["Gender"].fillna(2)

    values = {"No": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    x.loc[:, "University Degree"] = x["University Degree"].map(values)
    x.loc[:, "University Degree"] = x["University Degree"].fillna(0)

    values = {"Black": 0, "Brown": 1, "Red": 2, "Blond": 3}
    x.loc[:, "Hair Color"] = x["Hair Color"].map(values)
    x.loc[:, "Hair Color"] = x["Hair Color"].fillna(4)

    return x

def drop_columns(dataframe):
    """ Remove unnecessary input columns before inserting into model """
    return dataframe.drop(
        [
            "Hair Color",
            "Wears Glasses",
            "Body Height [cm]"
        ], 
        axis=1
    )

def split_data(dataframe):
    """ Splits data into training and test, also splits input from target"""
    train, test = train_test_split(dataframe, test_size=0.2)

    train_data = train.drop(["Income in EUR", "Instance"], axis=1)
    train_target = train[target_columns]

    test_data = test.drop(["Income in EUR", "Instance"], axis=1)
    test_target = test[target_columns]

    return (train_data, train_target, test_data, test_target)


if __name__ == "__main__":
    labelled_data = pandas.read_csv(with_labels)
    unlabelled_data = pandas.read_csv(without_labels)

    results = lin_model(labelled_data, unlabelled_data)
    results.to_csv(results_file, index=False)
