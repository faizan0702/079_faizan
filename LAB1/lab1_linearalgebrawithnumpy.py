
import numpy as np
import pandas as pd

matrix3x2 = np.array([[1, 2], [3, 4], [5, 6]])
matrix2x3 = np.array([[1, 2, 4], [4, 5, 6]])

print('3 x 2 matrix : \n', matrix3x2)
print('\n 2 x 3 matrix : \n', matrix2x3)

random_matrix = np.random.rand(3, 4)
print(random_matrix)

# matrix multiplication
matmul = np.dot(matrix3x2, matrix2x3)
print(matmul)

# elemet wise matrix multiplication
result = [[0 for x in range(len(matrix3x2))] for y in range(len(matrix2x3[0]))]

for i in range(len(matrix3x2)): 
  for j in range(len(matrix2x3[0])): 
    for k in range(len(matrix2x3)): 
      result[i][j] += matrix3x2[i][k] * matrix2x3[k][j]
 
print('Element wise multiplication : \n', result)

mean = np.mean(matrix3x2)
print(mean)
