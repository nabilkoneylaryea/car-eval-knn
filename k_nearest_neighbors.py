import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd. read_csv('./car.data')

# preprocessing to encode the data as numbers
label_encoder = preprocessing.LabelEncoder()

# TODO: is there a better way to do this??
# need to encode all the values as numerical values
buying = label_encoder.fit_transform(list(data['buying']))
maint = label_encoder.fit_transform(list(data['maint']))
door = label_encoder.fit_transform(list(data['door']))
persons = label_encoder.fit_transform(list(data['persons']))
lug_boot = label_encoder.fit_transform(list(data['lug_boot']))
safety = label_encoder.fit_transform(list(data['safety']))
cls = label_encoder.fit_transform(list(data['class']))

# labels we're predicting
predict = 'class'

# need lists specifically WHY (??)
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.2)

NUM_NEIGHBORS = 9 # hyperparameter for classifier

model = KNeighborsClassifier(n_neighbors=NUM_NEIGHBORS) # model with hyperparamter
model.fit(x_train, y_train) # training for model
acc = model.score(x_test, y_test) # testing for model

print('Accuracy:', acc)

predicted = model.predict(x_test) # make predictions by looping over test data
names = ['unacc', 'acc', 'good', 'vgood']

for index, data in enumerate(x_test):
    print('Predicted:', names[predicted[index]], 'Data:', x_test[index], 'Actual:', names[y_test[index]])
    neighbors = model.kneighbors([x_test[index]], 9, True)
    print('Neighbors:', neighbors)