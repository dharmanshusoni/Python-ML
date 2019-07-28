#hello 
#today we are going to make KNN clasification algo
# in IRIS dataset

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset["data"],
iris_dataset["target"],random_state=0)

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train,Y_train)

x_new = np.array([[5,2.7,1,0-8]])
#prediction method now ->

prediction = kn.predict(x_new)

print("Predict target Value -> {}\n".format(prediction))
print("Predict Feature Name -> {}\n".format(iris_dataset["target_names"][prediction]))
print("Score -> {:.2f}".format(kn.score(X_test,Y_test)))