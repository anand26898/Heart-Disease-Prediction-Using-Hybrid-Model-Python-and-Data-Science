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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def insert_probabilities(x, probabilities):
    new_x = []
    for row, prob in zip(x, probabilities):
        new_x.append(np.concatenate([row, prob]))

    return new_x


def random_forest(train_x, test_x, train_y, test_y):
    rf = RandomForestClassifier()
    rf.fit(train_x, train_y)
    prediction = rf.predict(test_x)
    return prediction


def decision_tree(train_x, test_x, train_y, test_y):
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x, train_y)
    prediction = clf.predict(test_x)
    return prediction


def hybrid(train_x, test_x, train_y, test_y):
	rf = RandomForestClassifier()
	rf.fit(train_x, train_y)
	probabilities = rf.predict_proba(train_x)
	train_x = insert_probabilities(train_x, probabilities)
	clf = tree.DecisionTreeClassifier()
	clf.fit(train_x, train_y)
	probabilities = rf.predict_proba(test_x)
	test_x = insert_probabilities(test_x, probabilities)
	y_pred = clf.predict(test_x)

	result2=open("results/resultHybrid.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()


	mse=abs(round(mean_squared_error(test_y, y_pred),2))
	mae=abs(round(mean_absolute_error(test_y, y_pred),2))
	r2=abs(round(r2_score(test_y, y_pred),2))
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR Hybrid Model IS %f "  % mse)
	print("MAE VALUE FOR Hybrid Model IS %f "  % mae)
	print("R-SQUARED VALUE FOR Hybrid Model IS %f "  % r2)
	rms = abs(round(np.sqrt(mean_squared_error(test_y, y_pred)),2))
	print("RMSE VALUE FOR Hybrid Model IS %f "  % rms)
	ac=round(accuracy_score(test_y,y_pred),2)
	print ("ACCURACY VALUE Hybrid Model IS %f" % ac)
	print("---------------------------------------------------------")

	result2=open('results/HybridMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/HybridMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  

	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('Hybrid Metrics Value')
	fig.savefig('results/HybridMetricsValueBarChart.png') 
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
	 
	plt.title('Hybrid Metrics Value')
	plt.savefig('results/HybridMetricsValue.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	cf = confusion_matrix(test_y, y_pred) 
	
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
	plt.savefig("results/HybridConfusion_Matrix.png") 
	plt.pause(5)
	plt.show(block=False)
	plt.close()	
	print('Report : ')
	print(classification_report(test_y, y_pred))

	return y_pred


def get_accuracy(test_y, prediction):
    correct = 0
    for actual, predicted in zip(test_y, prediction):
        if actual == predicted:
            correct += 1
    accuracy = str(correct / len(test_y) * 100)
    return accuracy


def process(path):
	dataset = pd.read_csv(path)
	X = dataset.iloc[:, 0:13].values
	y = dataset.iloc[:, 13].values

	train_x, test_x, train_y, test_y = train_test_split(X, y)
	
	hybrid_prediction = hybrid(train_x, test_x, train_y, test_y)
	hybrid_accuracy = get_accuracy(test_y, hybrid_prediction)
	
	dt_prediction = decision_tree(train_x, test_x, train_y, test_y)
	dt_accuracy = get_accuracy(test_y, dt_prediction)
	
	rf_prediction = random_forest(train_x, test_x, train_y, test_y)
	rf_accuracy = get_accuracy(test_y, rf_prediction)
	
	
	print("HYBRID:", hybrid_accuracy)
	print("DECISION TREE:", dt_accuracy)
	print("Random Forest:", rf_accuracy)


