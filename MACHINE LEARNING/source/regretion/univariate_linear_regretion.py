#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:31:37 2023 UTEC

@author: Cristian López Del Alamo
"""


import numpy as np
import matplotlib.pyplot as ptl 

class LinearRegresion:
    def __init__(self):
        self.m_W = np.random.random()
        self.m_b = np.random.random()
        
    def H(self,X):
        return self.m_W*X + self.m_b

    
    def Loss(self,X,Y):
        y_pred = self.H(X)
        return (np.linalg.norm((Y   - y_pred))**2)/(2*len(Y)), y_pred

    def dL(self,X,Y, Y_pre):
        dw =  np.sum((Y - Y_pre)*(-X))/len(Y)
        db =  np.sum((Y - Y_pre)*(-1))/len(Y)
        return dw,db 

    def change_params(self, dw,db, alpha):
        self.m_W = self.m_W - alpha*dw 
        self.m_b = self.m_b - alpha*db

    def train(self, X, Y, alpha,epochs):
        error_list = []
        time_stamp = []
        params = []
        for i in range(epochs):
            loss, y_pred = self.Loss(X,Y)
            time_stamp.append(i)
            error_list.append(loss)
            dw,db = self.dL(X,Y,y_pred)
            params.append([db,dw])
            self.change_params(dw,db,alpha)
            if(i%1000==0):
               # self.plot_error(time_stamp, error_list)
               print("error de pérdida : " + str(loss))
               self.plot_line(X,y_pred)
          
        return time_stamp, error_list,params
       

    def plot_error(self, time, loss):
       ptl.plot(time, loss)
       ptl.show()
  
    def plot_line(self,x,y_pre):
       ptl.plot(x, y,'*')
       ptl.plot(x,LR.H(x))
       ptl.show()
   
       
       
    

n = 100
epochs = 40000
 

x = np.array([i for i in range(n) ])
y = np.array([i + np.random.normal(0,0.4) for i in range(n) ])

x = (x - np.min(x)) / (np.max(x) - np.min(x))
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# rst = input("Quiere ver los resultados : (Y/Ny)")
LR = LinearRegresion()
time, loss,params = LR.train(x,y,0.001,epochs)

# rst = input("Quiere la aproxmación de la función : (Y/Ny)")
LR.plot_line(x, LR.H(x))
rst = input("Vericar el error de salida : (Y/N)")
LR.plot_error(time, loss)
print(params)

