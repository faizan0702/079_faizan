import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

data = pd.read_csv(r'C:\Users\faiza\OneDrive\Desktop\sem7\079_faizan\LAB3\PracticeDataSets\Dataset2.csv')
print(data)

# one hot
dummy_humidity = pd.get_dummies(data['Humidity'])
data = data.drop(['Humidity'],axis=1)
data = pd.concat([dummy_humidity,data],axis=1)

dummy_wind = pd.get_dummies(data['Wind'])
data = data.drop(['Wind'],axis=1)
data = pd.concat([dummy_wind,data],axis=1)

dummy_temp = pd.get_dummies(data['Temp'])
data = data.drop(['Temp'],axis=1)
data = pd.concat([dummy_temp,data],axis=1)

dummy_outlook = pd.get_dummies(data['Outlook'])
data = data.drop(['Outlook'],axis=1)
data = pd.concat([dummy_outlook,data],axis=1)

print("\n\nFinal Data :\n",data)

X = data.iloc[:, :-1] 

Y = data.iloc[:, -1] 


print("\n\nData : \n", X) 
print("\n\nTarget: \n", Y) 

from sklearn.model_selection import train_test_split

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(X,Y, test_size = 0.25, random_state = 79)

import numpy as np
from sklearn.naive_bayes import GaussianNB

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

q1 = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]
q2 = [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]

features = X.columns.tolist()
print(features)
df = pd.DataFrame([q1,q2], columns = features)
df.head()
output = gnb.predict(df)
print("Class predicted:\nq1: {}\nq2: {}".format(output[0],output[1]))

