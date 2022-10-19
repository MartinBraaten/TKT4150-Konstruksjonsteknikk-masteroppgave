'''
Created on 14. apr. 2021

@author: Martinskole
'''
'''
Created on 27. nov. 2020

@author: Martinskole
'''


import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
#from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt
from sympy import symbols, solve

data2 = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Nonknots') # load data set
data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Knots') # load data set


X2 = data.iloc[1: 19,0].values  # values converts it into a numpy array
Y2 = data.iloc[1: 19,1].values  # -1 means that calculate the dimension of rows, but have 1 column
X2 = np.array(X2)
Y2 = np.array(Y2)
X2 = X2.astype(float)
Y2 = Y2.astype(float)
print(X2,Y2)
print('len y =', len(Y2))
Char_val2 = (6.5*len(Y2)+6)/(3.7*len(Y2)-3)
print('Char_val2 =',Char_val2)

lin = linregress((X2, Y2))

A = lin[0]
B = lin[1]
R = lin[2]**2

A_inv = lin[0]**-1
B_inv = lin[1]*(-lin[0])**-1

xax = np.linspace(-1,11)

plt.plot(Y2, X2, 'r.', label = 'Measurements for specimen with knots')
plt.plot(lin[0]*xax + lin[1], xax, color='red',label = 'Regression curve for specimen with knots')


X2 = data2.iloc[1: 12,0].values  # values converts it into a numpy array
Y2 = data2.iloc[1: 12,1].values  # -1 means that calculate the dimension of rows, but have 1 column

X2 = np.array(X2)
Y2 = np.array(Y2)
print(X2,Y2)
X2 = X2.astype(float)
Y2 = Y2.astype(float)

print('len y =', len(Y2))
Char_val2 = (6.5*len(Y2)+6)/(3.7*len(Y2)-3)
print('Char_val2 =',Char_val2)

###############################################
lin2 = linregress((X2, Y2))

A = lin2[0]
B = lin2[1]
R = lin2[2]**2

A_inv = lin2[0]**-1
B_inv = lin2[1]*(-lin2[0])**-1

plt.plot(Y2, X2, 'b.', label = 'Measurements for specimen without knots')
plt.plot(lin2[0]*xax + lin2[1], xax, color='blue',label = 'Regression curve for specimen without knots')

p = symbols('p')
expr = lin2[0]*p + lin2[1] - (lin[0]*p + lin[1])
sol = solve(expr)

los = sol[0]
print('y =',los)
print('x =',lin2[0]*los + lin2[1])

plt.plot(-0.6,1,'bx',label = r'$f_{u}$ spruce')
plt.ylim((0, 1.4))  #0, 1.1 # 0, 1.6
plt.xlim((-1, 11))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Spruce: \n Knots vs without knots')
plt.legend()
plt.show()
