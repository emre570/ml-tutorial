#Importing our libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

#Getting our data
df = pd.read_csv("data\student\student-mat.csv", sep=";")
#df.head()

#We have too much unnecessary column. Getting the necessary ones.
data = df[["G1", "G2", "G3", "failures", "studytime", "absences"]]
#data.head()

'''Seperate our arrays to x and y.
    What this basically is; x is our features that we train the machine,
    y is what label are machine will predict.'''
x = np.array(data.drop(["G3"], 1))
y = np.array(data["G3"])

#Splitting our arrays as X train-test, Y train-test.
xtr, xte, ytr, yte = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
lnreg = sklearn.linear_model.LinearRegression()
lnreg.fit(xtr, ytr)

#acc for accuracy.
acc = lnreg.score(xte, yte)
print(acc)
#Getting our Coefficent and Intercept values
print('Coefficent: ', lnreg.coef_)
print('Intercept: ', lnreg.intercept_)

#Getting predictions based from our test array
pred = lnreg.predict(xte)

#Printing all of prediction data
for x in range(len(pred)):
    print(pred[x], xte[x], yte[x])

'''For instance, one of our output is: 13.940 [11 14 0 1 6] 14.
    What this actually is, first 13.940 is our machine's prediction based on:
    [11 first score, 14 second score (max 20), 0 failures, ,1 study time, 6 absences]
    Accuracy rate was %71 when this text is prepared.'''
