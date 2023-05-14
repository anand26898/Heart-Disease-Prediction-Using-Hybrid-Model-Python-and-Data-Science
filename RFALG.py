import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn import linear_model
#from sklearn.model_selection cross_validation
from scipy.stats import norm

from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



def process(path):
	dataset = pd.read_csv(path)
	X = dataset.iloc[:, 0:13].values
	y = dataset.iloc[:, 13].values

	X_train, X_test, y_train, y_test = train_test_split(X, y)

	model2=RandomForestClassifier()
	model2.fit(X_train, y_train)
	y_pred = model2.predict(X_test)
	print("predicted")
	print(y_pred)
	print("test")
	print(y_test)

	result2=open("results/resultRF.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()
	
	mse=abs(round(mean_squared_error(y_test, y_pred),2))
	mae=abs(round(mean_absolute_error(y_test, y_pred),2))
	r2=abs(round(r2_score(y_test, y_pred),2))
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR RandomForest IS %f "  % mse)
	print("MAE VALUE FOR RandomForest IS %f "  % mae)
	print("R-SQUARED VALUE FOR RandomForest IS %f "  % r2)
	rms = abs(round(np.sqrt(mean_squared_error(y_test, y_pred)),2))
	print("RMSE VALUE FOR RandomForest IS %f "  % rms)
	ac=round(accuracy_score(y_test,y_pred),2)
	print ("ACCURACY VALUE RandomForest IS %f" % ac)
	print("---------------------------------------------------------")
	

	result2=open('results/RFMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/RFMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title(' Random Forest Metrics Value')
	fig.savefig('results/RFMetricsValueBarChart.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()


	group_names=['MSE', 'MAE','R2','RMSE','ACCURACY']
	group_size=acc
	subgroup_names=acc
	subgroup_size=acc
	 
	# Create colors
	a, b, c,d,e=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens,plt.cm.Oranges,plt.cm.Purples]
	 
	# First Ring (outside)
	fig, ax = plt.subplots()
	ax.axis('equal')
	mypie, _ = ax.pie(group_size, radius=1.0, labels=group_names, colors=[a(0.6), b(0.6), c(0.6),d(0.1),e(0.6)] )
	plt.setp( mypie, width=0.3, edgecolor='white')
	 
	## Second Ring (Inside)
	mypie2, _ = ax.pie(subgroup_size, radius=1.0-0.3, labels=subgroup_names, labeldistance=0.7, colors=[a(0.6), b(0.6), c(0.6),d(0.1),e(0.6)] )
	plt.setp( mypie2, width=0.4, edgecolor='white')
	plt.margins(0,0)
	 
	plt.title(' Random Forest Metrics Value')
	plt.savefig('results/RFMetricsValue.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()


	cf = confusion_matrix(y_test, y_pred) 
	
	print(cf)
	
	
	plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
	plt.colorbar()
	plt.title('Confusion Matrix')
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	class_labels = ['0','1']
	tick_marks = np.arange(len(class_labels)) # length of classes
	tick_marks
	plt.xticks(tick_marks,class_labels)
	plt.yticks(tick_marks,class_labels)
	# plotting text value inside cells
	thresh = cf.max() / 2.
	for i in range(cf.shape[0]):
		for j in range(cf.shape[1]):
			plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
	plt.savefig("results/RFConfusion_Matrix.png") 
	plt.pause(5)
	plt.show(block=False)
	plt.close()	
	print('Report : ')
	print(classification_report(y_test, y_pred))

	