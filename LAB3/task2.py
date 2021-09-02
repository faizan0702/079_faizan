from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_wine

wine = load_wine()

print("Features: ", wine.feature_names)

print("Labels: ", wine.target_names)

# wine.data.shape
from sklearn.model_selection import train_test_split

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(wine.data,
                        wine.target, test_size = 0.20, random_state = 79)


import numpy as np
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(data_train, target_train)

#Predict the response for test dataset
target_pred = gnb.predict(data_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(target_test, target_pred))

#Import confusion_matrix from scikit-learn metrics module for confusion_matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(target_test, target_pred)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision = precision_score(target_test, target_pred, average=None)
recall = recall_score(target_test, target_pred, average=None)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))

