import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('Autism-Child-Data.csv')
to_drop = ['age','ethnicity','used_app_before', 'jundice','contry_of_res','result','age_desc', 'relation']

dataset.drop(columns=to_drop,inplace=True, axis=1)
dataset = dataset.replace({'Class/ASD': {'YES': 1, 'NO': 0}})
dataset = dataset.replace({'used_app_before': {'yes': 1, 'no': 0}})
dataset = dataset.replace({'gender': {'m': 1, 'f': 0}})

X1=dataset.drop('Class/ASD', axis=1)
X = dataset.iloc[:,:-1].values
y = dataset['Class/ASD'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = Sequential()
model.add(Dense(12, input_dim=11, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


