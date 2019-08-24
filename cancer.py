#pip install jupyter

import sklearn
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

lable_names = data['target_names']
lables = data['target']
feature_names = data['feature_names']
feature = data['data']

#Organising the data into sets

from sklearn.model_selection import train_test_split

train,test, train_labels,test_labels = train_test_split(feature,lables,test_size=0.33,random_state=42)

# build the model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

model = gnb.fit(train,train_labels)

prediction = gnb.predict(test)

# importing accuracy measuring function
from sklearn.metrics import accuracy_score

print(accuracy_score(test_labels,prediction))