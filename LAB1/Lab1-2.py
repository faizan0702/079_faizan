import numpy as np

alist = [1,2,3,4,5]
narray = np.array([1,2,3,4])

print(alist)
print(narray)

print(type(alist))
print(type(narray))

print(narray + narray)
print(alist + alist)

print(narray * 3)
print(alist * 3)

npmatrix1 = np.array([narray , narray ,narray])
npmatrix2 = np.array([narray , narray ,narray])
npmatrix3 = np.array([narray , narray ,narray])

print(npmatrix1)
print()

okmatrix = np.array([[1,2] , [3,4]])
print(okmatrix)
print(okmatrix *2)

print()
# badmatrix = np.array([[1,2],[3,4],[5,6,7]])
# print(badmatrix)
# print(badmatrix *3)




# Scalling and translation

result  = okmatrix *2 +1
print(result)

result1 = okmatrix + okmatrix 
print(result1)

result2 = okmatrix  - okmatrix
print(result2)


matrix3x2 = np.array([[1,2] , [3,4] , [5,6]])
print('original matrix 3 X 2')
print(matrix3x2)
print('transpose matrix 2 X 3')
print(matrix3x2.T)


# transpose operation does not effect on 1D array
nparray = np.array([1,2,3,4])
print('Transposed arraye')
print(nparray.T)

nparray = np.array([[1,2,3,4]])
print('original array')
print(nparray)
print('Transported array')
print(nparray.T)

nparray = np.array([1,2,3,4])
norm1 = np.linalg.norm(nparray)

nparray2 = np.array([[1,2],[3,4]])
norm2 = np.linalg.norm(nparray2)

print(norm1)
print(norm2)

nparray2 = np.array([[1,1],[2,2],[3,3]])
normByCols = np.linalg.norm(nparray2 , axis=0)
normByRows = np.linalg.norm(nparray2 , axis=1)

print(normByCols)
print(normByRows)

nparray1 = np.array([0,1,2,3])
nparray2 = np.array([4,5,6,7])

flavor1 = np.dot(nparray1 , nparray2)
print(flavor1)

flavour2 = np.sum(nparray1 * nparray2)
print(flavour2)

flavour3 = nparray1 @ nparray2
print(flavour3)

flavour4 = 0
for a,b in zip(nparray1 ,nparray2):
    flavour4 += a*b
print(flavour4)

norm1 = np.dot(np.array([1,2]) , np.array([3,4]))
norm2 = np.dot([1,2] , [3,4])

print(norm1 ,'=' , norm2)


# sum by rows or column
nparray2 = np.array([[1,-1] , [2,-2] , [3,-3]])

sumByCols = np.sum(nparray2 , axis=0)
sumByrows = np.sum(nparray2 , axis=1)

print(sumByCols)
print(sumByrows)

# get mean by row or col
nparray2 = np.array([[1,-1] , [2,-2] , [3,-3]])

mean = np.mean(nparray2)
meanByCol = np.mean(nparray2 , axis=0)
meanByRow = np.mean(nparray2 , axis=1)

print(mean)
print(meanByCol)
print(meanByRow)


# center column of matrix
nparray2 = np.array([[1,1],[2,2],[3,3]])
nparrayCentered = nparray2 - np.mean(nparray2 , axis=0)

print(nparray2)
print(nparrayCentered)