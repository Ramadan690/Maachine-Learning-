import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv('auto.csv')
X = data.drop('weight',axis=1)
y =  data['weight'] 

X_train , X_test , y_train, y_test = train_test_split(X, y, test_size = 0.10)

svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train,y_train)


y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
