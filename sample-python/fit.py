# Fit a simple linear regression model to the
# classic iris flower dataset to predict petal
# width from petal length. Write the fitted
# model to the file model.pkl

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import cdsw

import matplotlib.pyplot as plt


iris = datasets.load_iris()
test_size = 20

# Train
iris_x = iris.data[test_size:, 2].reshape(-1, 1) # petal length
iris_y = iris.data[test_size:, 3].reshape(-1, 1) # petal width

model = linear_model.LinearRegression()
model.fit(iris_x, iris_y)

# Test and predict
score_x = iris.data[:test_size, 2].reshape(-1, 1) # petal length
score_y = iris.data[:test_size, 3].reshape(-1, 1) # petal width

predictions = model.predict(score_x)

# Mean squared error
mean_sq = mean_squared_error(score_y, predictions)
cdsw.track_metric("mean_sq_err", mean_sq)
print("Mean squared error: %.2f"% mean_sq)

# Explained variance
r2 = r2_score(score_y, predictions)
cdsw.track_metric("r2", r2)
print('Variance score: %.2f' % r2)

# Output
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
cdsw.track_file(filename)
