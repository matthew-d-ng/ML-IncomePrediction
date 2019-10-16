# ML-IncomePrediction
Using Python scikit-learn to try fit a linear regression model 
for some sample data.
Results CSV is in the results folder. The file outside the results folder is the original empty one.

Dataset categories:
- Input data: "Year of Record", "Gender", "Age", "Country", "Size of City", "Profession", "University Degree", "Wears Glasses", "Hair Color", "Body Height [cm]"

- Target data: "Income in EUR"

Current pipeline:

- NaN and None values are removed from the training set.

- Categorical data is target encoded.

- Data is split using an 80/20 ratio into training and testing data.

- Values are scaled using sklearn's StandardScaler.

- Feature validation is recursively carried out using RFECV, with a Lasso model which examines error, with cross validation set to 5.

- The training data is then fitted to the target data using a KNeighborRegressor, taking into account 11 nearest neighbors for the vote, which is weighted according to distance.

- We then insert our testing data into the model for prediction, and proceed to compare it against our test targets. The root mean squared error is calculated, as well as "explained variance".


