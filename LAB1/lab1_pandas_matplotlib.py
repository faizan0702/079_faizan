import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\faiza\OneDrive\Desktop\sem7\079_faizan\LAB1\Data_for_Transformation.csv')

print(data.head())
plt.scatter(data["Age"], data['Salary'])
plt.show()

plt.hist(data['Salary'], bins = 10, color='green')
plt.show()

fig_size = plt.figure(figsize=(8, 5))
plt.bar(data['Country'], data['Salary'], color='yellow')
plt.xlabel('Salaries')
plt.ylabel('Countries')
plt.title('Bar chart of Country vs Salary')
plt.show()