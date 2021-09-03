
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\faiza\OneDrive\Desktop\sem7\079_faizan\LAB6\BuyComputer.csv')
data.drop(columns=['User ID',], axis = 1, inplace = True)
data.head()

#Declare label as last column in the source file
label = data.iloc[:,-1].values
# print(label)

#Declare X as all columns excluding last
x = data.iloc[:,:-1].values
# print('\n', x)

#splitting data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test   = train_test_split(x, label, test_size = 0.40, random_state = 79)
# print(x_train, '\n', x_test, '\n\n', y_train, '\n', y_test)

#scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#variables to calculate sigmoid function
y_pred = []
x_length = len(x_train[0])
w = []
b = 0.2
print(x_length)

entries = len(x_train[:, 0])
print(entries)


for weight in range(x_length):
  w.append(0)
print(w)

#sigmoid function
def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def predict(inputs):
    z = np.dot(w, inputs) + b
    temp = sigmoid(z)
    return temp

#Loss function
def loss_func(y, a):
    J = -(y * np.log(a) + (1-y) * np.log(1-a))
    return J


dw = []
db = 0
J = 0
alpha = 0.1
for x in range(x_length):
    dw.append(0)

#Repeating the process 3000 times
for iter in range(3000):
    for i in range(entries):
        local_x = x_train[i]
        a = predict(local_x)   
        dz = a - y_train[i]
        J += loss_func(y_train[i],a)

        for j in range(x_length):
            dw[j] = dw[j] + (local_x[j] * dz)

        db += dz
    J = J / entries
    db = db / entries

    for x in range(x_length):
        dw[x] = dw[x] / entries

    for x in range(x_length):
        w[x] = w[x] - (alpha * dw[x])
    b = b - (alpha * db)         
    J=0

#Print weight
print(w)

#print bias
print('\n', b)

#predicting the label
for x in range(len(y_test)):
    y_pred.append(predict(x_test[x]))

print("Actual\t\tPredicted")
for x in range(len(y_pred)):
    print(y_test[x] ,y_pred[x], sep="\t\t")
    if(y_pred[x] >= 0.5):
        y_pred[x] = 1
    else:
        y_pred[x] = 0


# Calculating accuracy of prediction
cnt = 0
for x in range(len(y_pred)):
    if(y_pred[x] == y_test[x]):
        cnt += 1

print('Accuracy  :  ', (cnt / (len(y_pred))) * 100)



# using sklearn logistic regression model
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 90)

#Fit
lr.fit(x_train, y_train)

#predicting the test label with lr. Predict always takes X as input
y_predict = lr.predict(x_test)

from sklearn.metrics import accuracy_score
print('Accuracy  :  ', accuracy_score(y_predict, y_test))


