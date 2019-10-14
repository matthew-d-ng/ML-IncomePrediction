# ML-IncomePrediction
Using Python scikit-learn to try fit a linear regression model 
for some sample data.

Dataset categories:
- Input data: "Year of Record", "Gender", "Age", "Country", "Size of City", "Profession", "University Degree", "Wears Glasses", "Hair Color", "Body Height [cm]"

- Target data: "Income in EUR"

Current pipeline:

- NaN and None values are removed from the training set.

- Categorical data is one hot encoded, with the exception of University Degree, which is encoded as a scale.

- Data is split using an 80/20 ratio into training and testing data.

- Values are scaled using sklearn's StandardScaler.

- The training data is then fitted to the target data using a LassoCV model, with cross-validation set to 5.

- We then insert our testing data into the model for prediction, and proceed to compare it against our test targets. The root mean squared error is calculated.


