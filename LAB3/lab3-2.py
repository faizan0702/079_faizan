import numpy as np 
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()

print("features :" , iris.feature_names)

print("labels :" ,iris.target_names)

print("\ndata shape:" , iris.data.shape)

print("\nTarget shape ",iris.target.shape)

print("data types :" , type(iris.data))

newdata = iris.data[50:,:]
newtarget = iris.target[50:]

print("\nnew data shape:",newdata.shape)
print("\nnew target shape:" , newtarget.shape)

from sklearn.model_selection import train_test_split
data_train , data_test , target_train , target_test = train_test_split(newdata,newtarget , test_size=0.30 , random_state = 5)

import numpy as np
gnb = GaussianNB()

gnb.fit(data_train , target_train)

target_pred = gnb.predict(data_test)

from sklearn import metrics
print("accuracy:" , metrics.accuracy_score(target_test , target_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix(target_test , target_pred)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision = precision_score(target_test , target_pred)
recall = recall_score(target_test , target_pred)

print("precision:{}".format(precision))
print('recall :{}'.format(recall))
