import numpy as np
import pandas as pd
from sklearn.svm import SVR

df=pd.read_csv('final_results.csv')
print(df)

X = df.drop('bmi','Height','Weight','Name','path').values
y = df['bmi'].values

clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X, y) 