from sklearn import linear_model as lm
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

def simple_model(labelled_data, unlabelled_data):
    """ Replaces NaN with -1, ignores string fields,
        does not modify data in any other way,
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


if __name__ == "__main__":
    labelled_data = pandas.read_csv(with_labels)
    unlabelled_data = pandas.read_csv(without_labels)

    results = simple_model(labelled_data, unlabelled_data)
    print(results)
    results.to_csv(results_file, index=False)
