'''
Created on 23. mar. 2021

@author: Martinskole
'''


import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt
 
data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='R=-1') # load data set
data1 = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Furu plot') # load data set

# rette opp her!
X2 = data.iloc[1: 30,5].values  # values converts it into a numpy array
Y2 = data.iloc[1: 30,4].values  # -1 means that calculate the dimension of rows, but have 1 column
X2 = np.array(X2)
Y2 = np.array(Y2)
X2 = X2.astype(float)
Y2 = Y2.astype(float)

start = 1
slutt = 33
Y1 = data1.iloc[start: slutt, 2].values  # values converts it into a numpy array
X1 = data1.iloc[start: slutt, 1].values  # -1 means that calculate the dimension of rows, but have 1 column
X1 = np.array(1.31718588859316*X1)
Y1 = np.array(Y1)
X1 = X1.astype(float)
Y1 = Y1.astype(float)

print(X2, Y2)
print('')
print(X1,Y1)

lin1 = linregress((X1, Y1))

A1 = lin1[0]
B1 = lin1[1]
R1 = lin1[2]**2

A_inv1 = lin1[0]**-1
B_inv1 = lin1[1]*(-lin1[0])**-1


lin = linregress((X2, Y2))

A = lin[0]
B = lin[1]
R = lin[2]**2

A_inv = lin[0]**-1
B_inv = lin[1]*(-lin[0])**-1


print('A_g=',A_inv, 'B_g=',B_inv, 'R2_g=',R)
print('A_f=',A_inv1, 'B_f=',B_inv1, 'R2_f=',R1)
y_mean = sum(Y2)/float(len(Y2))
SStot = sum( [(x - y_mean)**2 for x in Y2])
SSres = sum( [(y - (A*x + B))**2 for x, y in zip(X2, Y2) ])
R2 = 1 - SSres/SStot

N = -B_inv/A_inv
N1 = -B_inv1/A_inv1

N_halvsykel = 0.251189
Static_strength = A_inv*np.log10(N_halvsykel)+B_inv
Static_strength1 = A_inv1*np.log10(N_halvsykel)+B_inv1


print('N_g = ',N)
print('N_f = ',N1)

print('ss_g = ', (Static_strength)*100,'%', (Static_strength-1)*100,'%')
if Static_strength1<1:
    print('ss_f = ', (Static_strength1)*100, '%,', (1-Static_strength1)*100, '%')
else:
    print('ss_f = ', (Static_strength1)*100, '%,',(Static_strength1-1)*100, '%')



s = ((len(Y2)-2))**(-1) * SSres
std = sqrt(s)

SSres1 = sum( [(y - (A*x + B))**2 for x, y in zip(X1, Y1) ])
s1 = ((len(Y1)-2))**(-1) * SSres1
std1 = sqrt(s1)

print('s_g =', std)
print('s_f =', std1)

xax = np.linspace(-1,11)

plt.plot(Y2, X2, 'b.', label = 'Measurements spruce')
plt.plot(Y1,X1, 'r.', label = 'Measurements pine')
plt.plot(lin[0]*xax + lin[1], xax, color='blue',label = 'Regression curve spruce')
plt.plot(lin1[0]*xax + lin1[1], xax, color='red',label = 'Regression curve pine')
plt.plot(-0.6,1,'bx',label = r'$f_{u}$ spruce')
plt.plot(-0.6,1.31718588859316,'rx',label = r'$f_{v}$ pine')
plt.ylim((0, 1.4))  #0, 1.1
plt.xlim((-1, 10))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('S-N curve based on fatigue loading')
plt.legend()
plt.show()

