"""
K Nearest Neighbors Classification using Scikit Learn

K Nearest Neighbors Classifier separates data into classes by comparing it to the surrounding data and grouping the "nearest neighbors" together.
In the prediction step, the 
1. Separate training and testing data.
2. Create the model and specify the number of neighbors. K is the number of neighbors. Scikit learn uses the descriptive variable name n_neighbors.
3. Fit the model to the training data.
4. Predict using the testing data
   - Say we have a new unseen single test point (in Scikit learn a single row in X_test) and you want to give it a class label, i.e. you want to know which class this data point belongs to.
   - It goes through the training data features (all rows in X_train) and looks for the K closest neighbors (finds the K rows in X_train that are closest to the test point).
   - The class-label, or target, that appears the most in the K closest neighbors will be assigned to that test point.
   - Repeat the above process for all test points.
5. Test for accuracy and graph the data.
It is different from K

See also:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
"""

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

#data
X_train = np.array([[1,0], [1,2], [0,2], [5,4], [5,6]]) # training features
y_train = [0, 0, 0, 1, 1] # class labels also called targets

X_test = np.array([[0,1], [4,5], [2,2], [4,4]]) # unseen new features
y_test = [0, 1 ,0, 1] # answers we are trying to predict (we aren't supposed to know them)

#algorithm
#model = MLPClassifier(hidden_layer_sizes=(2,), solver='sgd', learning_rate_init=0.1, max_iter=1000)
model = KNeighborsClassifier(n_neighbors=1)

#training
model.fit(X_train, y_train)

#testing/prediction
y_pred = model.predict(X_test)

#analysis/plotting
acc = accuracy_score(y_test, y_pred)
print(acc)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='cool', label='Training data')
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap='cool', marker='x', label='Testing data')
plt.colorbar()
plt.legend()
