import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt

import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
#import NeuralNetwork as NN
#import predict as pred
#from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tkinter import filedialog
import preprocess as pre

import RFALG as RF
import DTALG as DT
import Hybrid as hy
import predict as pr

def clear():
    print("Clear1")
    txt.delete(0, 'end') 
    txt1.delete(0, 'end')    
    txt2.delete(0, 'end')
    txt3.delete(0, 'end')
    txt4.delete(0, 'end')   
    txt5.delete(0, 'end')   
    txt6.delete(0, 'end')   
    txt7.delete(0, 'end')   
    txt8.delete(0, 'end')   
    txt9.delete(0, 'end')   
    txt10.delete(0, 'end')   
    txt11.delete(0, 'end')   
    txt12.delete(0, 'end')   
    txt13.delete(0, 'end')   
    txt14.delete(0, 'end')   



window = tk.Tk()
window.title("Heart Disease Prediction")

 
window.geometry('1280x720')
bgcolor="#ffe6e6"
bgcolor1="#e60000"
fgcolor="#660000"

window.configure(background="#ffe6e6")
#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


message1 = tk.Label(window, text="Heart Disease Prediction" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=2,font=('times', 30, 'italic bold underline')) 
message1.place(x=100, y=20)

lbl = tk.Label(window, text="Data Set",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl.place(x=350, y=150)

txt = tk.Entry(window,width=15,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt.place(x=600, y=150)


lbl1 = tk.Label(window, text="Age",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl1.place(x=50, y=200)

txt1 = tk.Entry(window,width=15,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt1.place(x=300, y=200)

lbl2 = tk.Label(window, text="Sex",width=15  ,fg=fgcolor  ,bg=bgcolor    ,height=1 ,font=('times', 15, ' bold ')) 
lbl2.place(x=50, y=250)

txt2 = tk.Entry(window,width=15  ,bg=bgcolor  ,fg=fgcolor,font=('times', 15, ' bold ')  )
txt2.place(x=300, y=250)

lbl3 = tk.Label(window, text="CP",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl3.place(x=50, y=300)

txt3 = tk.Entry(window,width=15,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt3.place(x=300, y=300)

lbl4 = tk.Label(window, text="tresp",width=15  ,fg=fgcolor  ,bg=bgcolor    ,height=1 ,font=('times', 15, ' bold ')) 
lbl4.place(x=50, y=350)

txt4 = tk.Entry(window,width=15  ,bg=bgcolor  ,fg=fgcolor,font=('times', 15, ' bold ')  )
txt4.place(x=300, y=350)

lbl5 = tk.Label(window, text="chol",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl5.place(x=50, y=400)

txt5 = tk.Entry(window,width=15,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt5.place(x=300, y=400)

lbl6 = tk.Label(window, text="fbs",width=15  ,fg=fgcolor  ,bg=bgcolor    ,height=1 ,font=('times', 15, ' bold ')) 
lbl6.place(x=50, y=450)

txt6 = tk.Entry(window,width=15  ,bg=bgcolor  ,fg=fgcolor,font=('times', 15, ' bold ')  )
txt6.place(x=300, y=450)


lbl7 = tk.Label(window, text="restecg",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl7.place(x=50, y=500)

txt7 = tk.Entry(window,width=15,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt7.place(x=300, y=500)


lbl8 = tk.Label(window, text="thalach",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl8.place(x=600, y=200)

txt8 = tk.Entry(window,width=15,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt8.place(x=850, y=200)

lbl9 = tk.Label(window, text="exang",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl9.place(x=600, y=250)

txt9 = tk.Entry(window,width=15,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt9.place(x=850, y=250)

lbl10 = tk.Label(window, text="oldpeak",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl10.place(x=600, y=300)

txt10 = tk.Entry(window,width=15,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt10.place(x=850, y=300)


lbl11 = tk.Label(window, text="slope",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl11.place(x=600, y=350)

txt11 = tk.Entry(window,width=15,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt11.place(x=850, y=350)

lbl12 = tk.Label(window, text="ca",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl12.place(x=600, y=400)

txt12 = tk.Entry(window,width=15,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt12.place(x=850, y=400)

lbl13 = tk.Label(window, text="thal",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl13.place(x=600, y=450)

txt13 = tk.Entry(window,width=15,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt13.place(x=850, y=450)

lbl14 = tk.Label(window, text="Predicted Value",width=15  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl14.place(x=600, y=500)

txt14 = tk.Entry(window,width=25,bg=bgcolor ,fg=fgcolor,font=('times', 15, ' bold '))
txt14.place(x=850, y=500)


elbl1 = tk.Label(window, text="Ex:60",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl1.place(x=500, y=200)

elbl2 = tk.Label(window, text="0-M/1-F",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl2.place(x=500, y=250)

elbl3 = tk.Label(window, text="Ex:1-4",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl3.place(x=500, y=300)

elbl4 = tk.Label(window, text="Ex:90-180",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl4.place(x=500, y=350)

elbl5 = tk.Label(window, text="Ex:180-320",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl5.place(x=500, y=400)


elbl6 = tk.Label(window, text="Ex:0/1",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl6.place(x=500, y=450)


elbl7 = tk.Label(window, text="Ex:0/2",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl7.place(x=500, y=500)


elbl8 = tk.Label(window, text="Ex:90-200",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl8.place(x=1020, y=200)

elbl9 = tk.Label(window, text="Ex:0/1",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl9.place(x=1020, y=250)

elbl10 = tk.Label(window, text="Ex:0.0-4.0",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl10.place(x=1020, y=300)

elbl11 = tk.Label(window, text="Ex:1-3",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl11.place(x=1020, y=350)

elbl12 = tk.Label(window, text="Ex:0-3",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl12.place(x=1020, y=400)

elbl13 = tk.Label(window, text="Ex:3/6/7",width=7  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
elbl13.place(x=1020, y=450)


def browse():
	path=filedialog.askopenfilename()
	print(path)
	txt.delete(0, 'end')
	txt.insert('end',path)
	if path !="":
		print(path)
	else:
		tm.showinfo("Input error", "Select Dataset")	


def preprocess():
	path=txt.get()
	if path != "" :
		print("preprocess")
		pre.process(path)
		# read synthetic cleveland dataset from full cleveland.data
		tm.showinfo("Input error", "Preprocess Successfully Finished")
	else:
		tm.showinfo("Input error", "Select Dataset")
		
def RFprocess():
	sym=txt.get()
	if sym != "":
		RF.process(sym)
		tm.showinfo("Input", "RandomForest Successfully Finished")
	else:
		tm.showinfo("Input error", "Select Dataset")
		
def DTprocess():
	sym=txt.get()
	if sym != "":
		DT.process(sym)
		print("DT")
		tm.showinfo("Input", "DT Successfully Finished")
	else:
		tm.showinfo("Input error", "Select Dataset")
			
def hybridmodel():
	sym=txt.get()
	if sym != "":
		hy.process(sym)
		tm.showinfo("Input", "Hybrid Successfully Finished")
	else:
		tm.showinfo("Input error", "Select Dataset")
	
def predictprocess():
	print("predict")
	txt14.delete(0, 'end') 
	#txt1.insert('end', "60")
	a1=txt1.get()
	a2=txt2.get()
	a3=txt3.get()
	a4=txt4.get()
	a5=txt5.get()
	a6=txt6.get()
	a7=txt7.get()
	a8=txt8.get()
	a9=txt9.get()
	a10=txt10.get()
	a11=txt11.get()
	a12=txt12.get()
	a13=txt13.get()
	
	if a1 == "":
		tm.showinfo("Insert error", "Enter Age")
	elif a2 == "":
		tm.showinfo("Insert error", "Enter Sex")
	elif a3 == "":
		tm.showinfo("Insert error", "Enter Cp")
	elif a4 == "":
		tm.showinfo("Insert error", "Enter tresp")
	elif a5 == "":
		tm.showinfo("Insert error", "Enter Chol")
	elif a6 == "":
		tm.showinfo("Insert error", "Enter fbs")
	elif a7 == "":
		tm.showinfo("Insert error", "Enter restecg")
	elif a8 == "":
		tm.showinfo("Insert error", "Enter thalach")
	elif a9 == "":
		tm.showinfo("Insert error", "Enter exang")
	elif a10=="":
		tm.showinfo("Insert error", "Enter oldpeak")
	elif a11 == "":
		tm.showinfo("Insert error", "Enter slope")
	elif a12 == "":
		tm.showinfo("Insert error", "Enter ca")
	elif a13 == "":
		tm.showinfo("Insert error", "Enter thal")
	else:
		
		#new_pred = model.predict_classes(np.array([[58,1,3,112,230,0,2,165,0,2.5,2,1,7]]))  #4
		#new_pred = model.predict_classes(np.array([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13]]))
		new_pred=pr.process([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13])
		res=int(new_pred[0])
		print(res)
		if res == 0:
			print("sdsd")
			txt14.insert('end', "No Heart Disease")
		else:
			print("sds no")
			txt14.insert('end', "Heart Disease Possible")
		#nn=np.array([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13]])
		#print(nn)
		#pred.process(nn)

		
    

br = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1   ,width=10  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
br.place(x=800, y=140)

clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=10  ,height=1 ,activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=950, y=140)


process = tk.Button(window, text="Preprocess", command=preprocess  ,fg=fgcolor  ,bg=bgcolor1   ,width=17  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
process.place(x=50, y=600)

DTbutton = tk.Button(window, text="Decision Tree", command=DTprocess  ,fg=fgcolor  ,bg=bgcolor1  ,width=17  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
DTbutton.place(x=250, y=600)

RFbutton = tk.Button(window, text="Random Forest", command=RFprocess  ,fg=fgcolor  ,bg=bgcolor1   ,width=17  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
RFbutton.place(x=460, y=600)


HYbutton = tk.Button(window, text="Hybrid", command=hybridmodel ,fg=fgcolor  ,bg=bgcolor1   ,width=17  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
HYbutton.place(x=650, y=600)


predict = tk.Button(window, text="Predict", command=predictprocess  ,fg=fgcolor  ,bg=bgcolor1   ,width=17  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
predict.place(x=850, y=600)



quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor  ,bg=bgcolor1   ,width=17  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1060, y=600)

 
window.mainloop()


