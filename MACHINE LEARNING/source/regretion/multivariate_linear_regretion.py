#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:31:37 2023 UTEC

@author: Cristian López Del Alamo
"""



import numpy as np
import matplotlib.pyplot as ptl 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
class LinearRegresion:
    
    def __init__(self,dim):
        self.dim = dim
        self.m_W = np.random.rand(dim)
        self.m_b = np.random.random()
        
    def H(self,X):
        return np.dot(X,self.m_W) + self.m_b

    
    def Loss(self,X,Y):
        y_pred = self.H(X)
        return (np.linalg.norm((Y   - y_pred))**2)/(2*len(Y)), y_pred

    def dL(self,X,Y, Y_pre):
        dw =  np.matmul(Y - Y_pre,-X)/len(Y)
        db =  np.sum((Y - Y_pre)*(-1))/len(Y)
        return dw,db 

    def change_params(self, dw,db, alpha):
        self.m_W = self.m_W - alpha*dw 
        self.m_b = self.m_b - alpha*db

    def train(self, X, Y, alpha,epochs):
        error_list = []
        time_stamp = []
        for i in range(epochs):
            loss, y_pred = self.Loss(X,Y)
            time_stamp.append(i)
            error_list.append(loss)
            dw,db = self.dL(X,Y,y_pred)
            self.change_params(dw,db,alpha)
            # print("error de pérdida : " + str(loss))
            if(i%100==0):
            #    self.plot_error(time_stamp, error_list)
                 LR.plot_plane(X[:,0],X[:,1],Y)
            
        return time_stamp, error_list
       

    def plot_error(self, time, loss):
       ptl.plot(time, loss)
       ptl.show()
  
    def plot_line(self,x,y_pre):
       ptl.plot(x, y,'*')
       ptl.plot(x,LR.H(x))
       ptl.show()
       
    def plot_plane(self,xx,yy,zz):
        fig = ptl.figure(figsize=(8, 8))
       
        
        # crear un subplot 3D
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xx, yy, zz, c='r', marker='o')
        
        # definir el tamaño del plano
        x = np.linspace(min(xx), max(xx),len(xx))  
        y = np.linspace(min(yy), max(yy),len(yy))
        
  
        x, y = np.meshgrid(x, y)
        
        # definir la ecuación del plano
        z = self.m_W[0]*x + self.m_W[1]*y+ self.m_b 
      
        
        # dibujar el plano
        ax.plot_surface(x, y, z )
        ax.plot_surface

        ax.view_init(elev=0, azim=70) 
        
        # agregar etiquetas a los ejes
        ax.set_xlabel('Feacture 1: X0')
        ax.set_ylabel('Feacture 2: X1')
        ax.set_zlabel('Prediction')
        
       
        ax.mouse_init()

        ptl.show()  
       



        


 
n = 400
epocas = 200
feactures = 2
x= np.zeros((n, feactures))
x[:, 0] = np.arange(1, n+1) + np.random.rand(n)*10
x[:, 1] = np.arange(1, n+1) + np.random.rand(n)*10
# x[:, 2] = np.arange(1, n+1)
# x[:, 3] = np.arange(1, n+1)
# x[:, 4] = np.arange(1, n+1)

columnas = np.random.uniform(-1, 1, size=(n, feactures))
x+=columnas

y = np.array([i + np.random.normal(0,2.5) for i in range(n) ])

x = (x - np.min(x)) / (np.max(x) - np.min(x))
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# rst = input("Quiere ver los resultados : (Y/Ny)")
LR = LinearRegresion(2)
time, loss = LR.train(x,y,0.1,epocas)
LR.plot_error(time, loss)
# rst = input("Quiere la aproxmación de la función : (Y/Ny)")
rst = input("Vericar el error de salida : (Y/N)")
LR.plot_line(x, LR.H(x))

# LR.plot_plane(x[:,0],x[:,1],y)
