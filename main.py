from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

#data
X_train = np.array([[1,0],[1,2],[0,2],[5,4],[5,6]])
y_train = [0, 0, 0, 1, 1]

X_test = np.array([[0,1],[4,5],[2,2],[4,4]])
y_test = [0,1,0,1]

#algorithm
#model = MLPClassifier(hidden_layer_sizes=(2,), solver='sgd', learning_rate_init=0.1, max_iter=1000)
model = KNeighborsClassifier(n_neighbors=1)

#training
model.fit(X_train,y_train)

#testing/prediction
y_pred = model.predict(X_test)

#analysis/plotting
acc = accuracy_score(y_test, y_pred)
print(acc)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='cool')
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap='cool', marker='x')
plt.colorbar()
