import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def process(path):
	df_main = pd.read_table(path, sep=',')
	df_main.astype(float)
	# Normalize values to range [0:1]
	df_main /= df_main.max()
	# split data into independent and dependent variables
	y_all = df_main['num']
	X_all = df_main.drop(columns = 'num')
	fig, axs = plt.subplots(nrows=1, ncols=1, sharey=False, figsize=(10,5))
	axs.set_title('DataSet')
	axs.grid()
	axs.hist(y_all)
	axs.get_children()[0].set_color('g')
	axs.get_children()[2].set_color('c')
	axs.get_children()[5].set_color('b')
	axs.get_children()[7].set_color('y')
	axs.get_children()[9].set_color('r')
	fig.savefig('results/Preprocess.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	# read synthetic cleveland dataset from full cleveland.data
	data = pd.read_table(path, sep=',')
	

	names=list(data.columns)
	
	
	fig, ax = plt.subplots(figsize=(15,7))    	
	ncols=3
	plt.clf()
	f = plt.figure(1)
	f.suptitle(" Data Histograms", fontsize=12)
	vlist = list(data.columns)
	nrows = len(vlist) // ncols
	if len(vlist) % ncols > 0:
		nrows += 1
	for i, var in enumerate(vlist):
		plt.subplot(nrows, ncols, i+1)
		plt.hist(data[var].values, bins=15)
		plt.title(var, fontsize=10)
		plt.tick_params(labelbottom='off', labelleft='off')
	plt.tight_layout()
	plt.subplots_adjust(top=0.88)
	fig.savefig('results/Data Histograms.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()


	fig, ax = plt.subplots(figsize=(15,7))
	data[["num", "age"]].groupby(["age"]).count().plot.bar(stacked=True,ax=ax)
	print(data)
	ax.title.set_text('Number of Records in Age')
	ax.set_ylabel('Sum Value')
	plt.savefig('results/AgeCount.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()


	d=data[data.num==1].groupby(["age"]).count()
	df = pd.DataFrame(d)
	df1=df["num"]
	d=data[data.num==0].groupby(["age"]).count()
	df = pd.DataFrame(d)
	df2=df["num"]
	df=pd.concat([df1, df2], axis=1, sort=False)
	df=df.fillna(0.0)
	df.columns = [ 'y', 'n']
	print(df)
	fig, ax = plt.subplots(figsize=(15,7))
	df.plot.bar(stacked=True,ax=ax)
	ax.title.set_text('Age')
	ax.set_ylabel('Sum Value')
	plt.savefig('results/Age.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	fig, ax = plt.subplots(figsize=(15,7))
	data[["num", "sex"]].groupby(["sex"]).count().plot.bar(stacked=True,ax=ax)
	print(data)
	ax.title.set_text('Sex')
	ax.set_ylabel('Sum Count Per Class Value')
	plt.savefig('results/SexCount.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	d=data[data.num==1].groupby(["sex"]).count()
	df = pd.DataFrame(d)
	df1=df["num"]
	d=data[data.num==0].groupby(["sex"]).count()
	df = pd.DataFrame(d)
	df2=df["num"]
	df=pd.concat([df1, df2], axis=1, sort=False)
	df=df.fillna(0.0)
	df.columns = [ 'y', 'n']
	print(df)
	fig, ax = plt.subplots(figsize=(15,7))
	df.plot.bar(stacked=True,ax=ax)
	ax.title.set_text('Sex')
	ax.set_ylabel('Sum Value')
	plt.savefig('results/Sex.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()
	
	

#process("heartdata.csv")