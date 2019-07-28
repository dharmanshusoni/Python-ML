from sklearn import datasets, linear_model, metrics

digits = datasets.load_digits()

x = digits.data
y = digits.target

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.4,random_state=1)

reg = linear_model.LogisticRegression()

reg.fit(X_train,Y_train)

y_predic = reg.predict(X_test)

print("Logistic Reg model accuracy(in %)", metrics.accuracy_score(Y_test,y_predic)*100 )