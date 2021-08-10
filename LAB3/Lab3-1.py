from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB , MultinomialNB

weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy','Rainy', 'Overcast',
           'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']

temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild',
        'Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes',
      'Yes','Yes','Yes','Yes','No']

le = preprocessing.LabelEncoder()

weather_encoded = le.fit_transform(weather)
print('weather:' , weather_encoded)

temp_encoded = le.fit_transform(temp)
label = le.fit_transform(play)

print("temp:" ,temp_encoded)
print("play:" , label)

features = tuple(zip(weather_encoded , temp_encoded))
print("features:", features)

model = MultinomialNB()
model.fit(features , label)

#sunny->2 , overcast->0 , rainy->1 
# mild->2 , hot->1 , cold->0

predicted = model.predict([[1,0]])
print("pridicted value:" , predicted)

predicted = model.predict([[0 , 1]])
print("pridicted value:" , predicted)

predicted = model.predict([[2,2]])
print("pridicted value:" , predicted)

predicted = model.predict([[0,1]])
print("pridicted value:" , predicted)

predicted = model.predict([[2,2]])
print("pridicted value:" , predicted)

predicted = model.predict([[1,2]])
print("pridicted value:" , predicted)
