from numpy.lib.function_base import average
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

le = preprocessing.OneHotEncoder()

iris = datasets.load_iris()
# iris_encoded = le.fit_transform(iris)

# print(iris)
# print("feature :" , iris_encoded.feature_names)
# print("labels: " , iris_encoded.target_names)

iris.data.shape

from sklearn.model_selection import train_test_split

data_train , data_test , target_train , target_test = train_test_split(iris.data , iris.target,test_size=.025 , random_state=10)

import numpy as np

gnb = GaussianNB()

gnb.fit(data_train , target_train)

target_pred = gnb.predict(data_test)

from sklearn import metrics 
print("accuracy :" , metrics.accuracy_score(target_test , target_pred))

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision = precision_score(target_test , target_pred , average=None)
recall = recall_score(target_test , target_pred , average=None)

print('precision:.{}'.format(precision))

print('recall:.{}'.format(recall))