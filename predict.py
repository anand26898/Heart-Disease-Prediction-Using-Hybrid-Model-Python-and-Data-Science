import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def insert_probabilities(x, probabilities):
    new_x = []
    for row, prob in zip(x, probabilities):
        new_x.append(np.concatenate([row, prob]))

    return new_x




def hybrid(train_x, test_x, train_y):
	rf = RandomForestClassifier()
	rf.fit(train_x, train_y)
	probabilities = rf.predict_proba(train_x)

	train_x = insert_probabilities(train_x, probabilities)

	clf = tree.DecisionTreeClassifier()
	clf.fit(train_x, train_y)
	probabilities = rf.predict_proba(test_x)
	test_x = insert_probabilities(test_x, probabilities)
	y_pred = clf.predict(test_x)

	return y_pred




def process(test_x):
	test_x = np.array(test_x).reshape((1, -1))
	dataset = pd.read_csv("heartdata.csv")
	train_x = dataset.iloc[:, 0:13].values
	train_y = dataset.iloc[:, 13].values
	
	
	hybrid_prediction = hybrid(train_x, test_x, train_y)
	print(hybrid_prediction)
	return hybrid_prediction