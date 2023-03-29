#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:31:37 2023 UTEC

@author: Cristian López Del Alamo
"""


import numpy as np
import matplotlib.pyplot as ptl 

class LinearRegresion:
    def __init__(self, grado):
        self.m_W = np.random.rand(grado)
        self.m_b = np.random.random()
        self.grado = grado
      
        
        
    def H(self,X):
        return np.dot(X,self.m_W)
    # + self.m_b

    def predic(self,x):
       potencias = np.arange(self.grado)
       x = np.power.outer(x, potencias)
       return np.dot(x,self.m_W)
       

    
    def Loss(self,X,Y):
        y_pred = self.H(X)
        return (np.linalg.norm((Y   - y_pred))**2)/(2*len(Y)), y_pred 

    def dL(self,X,Y, Y_pre):
        dw =  np.matmul(Y - Y_pre,-X)/len(Y)
        db =  np.sum((Y - Y_pre)*(-1))/len(Y)
        return dw,db 

    def change_params(self, dw,db, alpha):
        self.m_W = self.m_W - alpha*dw 
        # self.m_b = self.m_b - alpha*db

    def train(self, X, Y, alpha,epochs):
        error_list = []
        time_stamp = []
        potencias = np.arange(self.grado)
        X = np.power.outer(X, potencias)
        
        for i in range(epochs):
            loss, y_pred = self.Loss(X,Y)
            time_stamp.append(i)
            error_list.append(loss)
            dw,db = self.dL(X,Y,y_pred)
            self.change_params(dw,db,alpha)
           
            if(i%10000==0):
               # self.plot_error(time_stamp, error_list)
               # print("error de pérdida : " + str(loss))
               LR.plot_line(x, LR.predic(x))
        return time_stamp, error_list
       

    def plot_error(self, time, loss):
       ptl.plot(time, loss)
       ptl.show()
  
    def plot_line(self,x,y_pre):
       ptl.plot(x, y,'.')
       ptl.plot(x,y_pre)
       ptl.show()


n = 100

x = np.linspace(0, 2*np.pi, n)
y = np.sin(x)

x = (x - np.min(x)) / (np.max(x) - np.min(x))
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# rst = input("Quiere ver los resultados : (Y/Ny)")
LR = LinearRegresion(100)
time, loss = LR.train(x,y,0.3,100000)
LR.plot_error(time, loss)
# # rst = input("Quiere la aproxmación de la función : (Y/Ny)")
LR.plot_line(x, LR.predic(x))
