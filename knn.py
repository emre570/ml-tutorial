import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("data\car.data")
#data.head()

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

xtr, xte, ytr, yte = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#7 neighbors giving the best accuracy.
model = KNeighborsClassifier(n_neighbors=7)
model.fit(xtr, ytr)
acc = model.score(xte, yte)
print(acc)
#Machine predicts the values
predict = model.predict(xte)
'''Let's see how it predicted and what the real values are.
Let's make something different.
Hold a mistake and total value and see our inaccuracy percentage too.
This inaccuracy stuff is added by me.'''
mistake = 0
total = len(predict)
for x in range(len(predict)):
    print("Predict:", predict[x], " Data:", xte[x], " Actual:", yte[x])
    if predict[x] != yte[x]:
        mistake+=1

inacc = (mistake / total)
print("Mistakes: ", mistake, " Inaccuracy:", inacc)