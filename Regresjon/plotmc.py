
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt
from sympy import symbols, solve



xax = np.linspace(-1,11)

#fatigue = [N,-0.6]
#karakt = [0,Kar]
#fatigue1 = [N-std*Char_val2,-0.6]

A = [-1.66,-2.74, -4.04, -7.13, -8.4 ]
B_1 = [99.5, 99.2, 98.8, 97.9, 97.5]
C =  [-0.6, -0.6, -0.6, -0.6, -0.6]

B= []
for i in range(len(A)):
    B.append(A[i]*(0.6)+B_1[i])
Y_0 = []
for i in range(len(A)):
    Y_0.append(-B[i]/A[i])
Xverdier = [-0.6, Y_0[0], -0.6, Y_0[1], -0.6,Y_0[2], -0.6, Y_0[3], -0.6, Y_0[4]]
Yverdier = [B_1[0],0,B_1[1],0,B_1[2],0,B_1[3],0,B_1[4],0,]
#for i in range(len(A)):
 #   plt.plot(A[i]*xax+B[i], color='black', label = 'Relative decline in characteristic value')
#plt.plot(Y2, X2, 'b.', label = 'Measurements')
#plt.plot(7,0.16,'+',color='black', label = 'Did not fail')
#plt.plot(-0.6,1,'bx',label = r'$f_{u}$')
#plt.plot(lin[0]*xax + lin[1], xax, color='red',label = 'Regression curve')
#plt.plot(lin[0]*xax + lin[1]-Char_val2*std,xax, color='green',label = 'Characteristic curve') #1.644854
for i in range(0, len(A)):
    plt.plot(Xverdier[i:i+2], Yverdier[i:i+2],color='black', label = 'Relative decline in characteristic value')
#for i in range(0, len(fatigue), 2):
#    plt.plot(fatigue1[i:i+2], karakt[i:i+2],color='blue', label = 'Most conservative for design')
plt.ylim((0, 110))  #0, 1.1 # 0, 1.3
plt.xlim((-1, 8))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Pine: \n S-N curve based on fatigue loading')
plt.legend()
plt.show()