

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns


dataset = pd.read_csv(r'C:\Users\faiza\OneDrive\Desktop\sem7\079_faizan\LAB2\Datasets\Exercise-CarData.csv', index_col=[0])
print('Data : \n', dataset)
print('\nData Statistics : \n', dataset.describe())

"""Applying Label Encoder on data to convert string labels into numeric values"""

dataset.dropna(how='all', inplace=True)
print(dataset.dtypes)

new_x = dataset.iloc[:, :-1].values #all rows except last
new_y = dataset.iloc[:, -1,].values  # last column

new_x[:, 3] = new_x[:, 3].astype('str')
le = LabelEncoder()
new_x[:, 3] = le.fit_transform(new_x[:, 3])

print('\nInput before imputation : \n', new_x[6])

"""Handling null values imputation (replacing null values with mean and mode values of that attribute"""

str_to_num_dictionary={"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}

#for col-3
for i in range(new_x[:, 3].size):
  if (new_x[i, 2] == "??") :
    new_x[i, 2] = np.nan
  
  if (new_x[i, 4] == "????") :
    new_x[i, 4] = np.nan #.
  
  temp = str(new_x[i, 8])
  if (temp.isnumeric()) :
    new_x[i, 8] = int(temp)
  else:
    new_x[i, 8] = str_to_num_dictionary[temp]

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
mode_imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

the_imputer = imputer.fit(new_x[:, 0:3]) #fitting the data, function learns the stats
new_x[:, 0:3] = the_imputer.transform(new_x[:, 0:3])

the_mode_imputer = mode_imputer.fit(new_x[:, 3:4]) #fitting the data, function learns the stats
new_x[:, 3:4] = the_mode_imputer.transform(new_x[:, 3:4])

the_imputer = imputer.fit(new_x[:, 4:5]) #fitting the data, function learns the stats
new_x[:, 4:5] = the_imputer.transform(new_x[:, 4:5])

the_mode_imputer = mode_imputer.fit(new_x[:, 5:6])   
new_x[:, 5:6] = the_mode_imputer.transform(new_x[:, 5:6])

print('\nNew Input with Mean Value for NaN : \n', new_x[6])

"""Converting numpy ndarray to pandas dataframe"""

new_dataset = pd.DataFrame(new_x, columns=dataset.columns[:-1])
new_dataset = new_dataset.astype(float)
new_dataset.dtypes

"""Feature Selection"""

corr = new_dataset.corr()
print("corelation  head \n" , corr.head())
sns.heatmap(corr)

cols = np.full((len(new_dataset.columns), ), True, dtype=bool)
for i in range(corr.shape[0]):
  for j in range(i+1, corr.shape[0]):
    if (corr.iloc[i, j] >= 0.9) :
      if (cols[j]):
        cols[j] = False

selected_cols = new_dataset.columns[cols]
print(selected_cols)

new_dataset = new_dataset[selected_cols]

"""Scaling and Transformation"""

new_x = new_dataset.iloc[:, :-1].values
scaler = MinMaxScaler()
std = StandardScaler()
new_x[:, 0:3] = std.fit_transform(scaler.fit_transform(new_x[:, 0:3]))
new_x[:, 4:5] = std.fit_transform(scaler.fit_transform(new_x[:, 4:5]))
new_x[:, 7:9] = std.fit_transform(scaler.fit_transform(new_x[:, 7:9]))

print('\nDataset after preprocessing : \n', new_dataset)