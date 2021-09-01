import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from subprocess import call


wine_data = datasets.load_wine()
ds = pd.DataFrame(wine_data.data, columns = wine_data.feature_names)
print(f"#examples :{ds.shape[0]} and #features: {ds.shape[1]}")


print(ds.head())
print("\n\nFeatures:", wine_data.feature_names)
print("\nLabels:", np.unique(wine_data.target_names))


# random state sa per roll no
x_train, x_test, y_train, y_test = train_test_split(wine_data.data, wine_data.target, test_size = 0.20, random_state = 79)

#creating instance of classifier and performing training
dtclassifier = DecisionTreeClassifier(criterion = "entropy", max_leaf_nodes = 10)
dtclassifier.fit(x_train,y_train)

# Testing
y_prediction = dtclassifier.predict(x_test)

#  Accuracy
accuracy = accuracy_score(y_test, y_prediction)
print("Accuracy Score:\n", accuracy)

#  Confusion Matrix
c_matrix = confusion_matrix(y_test, y_prediction)
print("\nConfusion Matrix:\n",c_matrix)

#  Precision
precision = precision_score(y_test, y_prediction, average=None)
print("\nPrecision Score:\n", precision)

#  Recall
recall = recall_score(y_test, y_prediction, average=None)
print("\nRecall Score:\n", recall)