from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
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
    "Gender", "Country", "Profession", "University Degree"
]

input_columns = [
    "Year of Record", "Gender", "Age",
    "Country", "Size of City", "Profession", "University Degree",
    "Wears Glasses", "Hair Color", "Body Height [cm]"
]

target_columns = [
    "Income in EUR"
]

def drop_columns(dataframe):
    """ Remove unnecessary input columns before inserting into model
    """
    return dataframe.drop(
        [
            "Hair Color",
            "Wears Glasses",
            "Body Height [cm]"
        ], 
        axis=1
    )

def bad_model(labelled_data, unlabelled_data):
    """ Replaces NaN values with -1, 
        ignores non-numeric fields,
        does not modify data in any other way.
    """
    # removing NaN fields
    clean_labelled = labelled_data.dropna() # so a -1 won't skew the model
    clean_unlabelled = unlabelled_data[no_str_columns] # income field is NaN
    clean_unlabelled = clean_unlabelled.fillna(-1) # don't want to remove any

    # specifying relevant columns
    training_data = clean_labelled[no_str_columns]
    target_data = clean_labelled[["Income in EUR"]]
    test_data = clean_unlabelled[no_str_columns]

    # building model
    lin_reg = lm.LinearRegression()
    lin_reg.fit(training_data, target_data)

    # getting results and formatting them to DataFrame
    values = lin_reg.predict(test_data)
    results = pandas.DataFrame({
        "Instance": clean_unlabelled["Instance"].values,
        "Income": values.flatten()
    })
    return results

def better_model(labelled_data, unlabelled_data):
    """ Replaces NaN values with column mean,
        categorises non-numeric fields with OneHotEncoder,
        80/20 splits data to help verify model,
        use LassoCV model
    """
    print("cleaning data...")
    clean_labelled = labelled_data.dropna()
    clean_unlabelled = unlabelled_data[all_columns]
    clean_unlabelled = clean_unlabelled.fillna(method="ffill")
    clean_unlabelled = clean_unlabelled.fillna("None")

    # clean input data
    clean_labelled = drop_columns(clean_labelled)
    clean_unlabelled = drop_columns(clean_unlabelled)

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
    train, test = train_test_split(clean_labelled, test_size=0.2)

    training_data = train.drop(["Income in EUR"], axis=1).drop(["Instance"], axis=1)
    target_data = train[target_columns]

    test_data = test.drop(["Income in EUR"], axis=1).drop(["Instance"], axis=1)
    test_target = test[target_columns]

    print("fitting model...")
    # fit model
    lasso = lm.LassoCV(cv=3)
    lasso.fit(training_data, target_data)

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
    return results

def encode_training(ohe, dataframe):
    x = constrain_col_vals(dataframe)
    arr = ohe.fit_transform(x[ohe_columns])
    x = merge_encoding(ohe, x, arr)
    return x

def encode_testing(ohe, dataframe):
    x = constrain_col_vals(dataframe)
    arr = ohe.transform(x[ohe_columns])
    x = merge_encoding(ohe, x, arr)
    return x

def merge_encoding(ohe, dataframe, arr):
    col_names = ohe.get_feature_names(ohe_columns)
    encoded = pandas.DataFrame(arr, columns = col_names)
    dataframe = dataframe.drop(ohe_columns, axis=1).reset_index(drop=True)
    dataframe = dataframe.join(encoded)
    return dataframe

def constrain_col_vals(dataframe):
    x = dataframe

    mapping = {"male": 0, "female": 1, "other": 2}
    x["Gender"] = x["Gender"].map(mapping)
    x["Gender"] = x["Gender"].fillna(2)

    mapping = {"No": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    x["University Degree"] = x["University Degree"].map(mapping)
    x["University Degree"] = x["University Degree"].fillna(0)

    return x


if __name__ == "__main__":
    labelled_data = pandas.read_csv(with_labels)
    unlabelled_data = pandas.read_csv(without_labels)

    # testing(labelled_data)
    results = better_model(labelled_data, unlabelled_data)
    results.to_csv(results_file)
