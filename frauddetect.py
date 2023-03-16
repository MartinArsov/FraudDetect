import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

df = pd.read_csv('fraud_data.csv')

df['amount'] = df['amount'].astype(float)
df['time'] = df['time'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(df[['amount', 'time']].values, df['class'].values, test_size=0.2, random_state=0)

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

f1 = f1_score(y_test, y_pred)
print('F1-Score:', f1)