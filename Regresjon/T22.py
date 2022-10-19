

import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
#from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt
from sympy import symbols, solve

data2 = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='R=-1') # load data set
data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='R=-1') # load data set


X2 = data.iloc[2: 10,58].values  # values converts it into a numpy array
Y2 = data.iloc[2: 10,60].values  # -1 means that calculate the dimension of rows, but have 1 column
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
print('A=',A_inv, 'B=',B_inv, 'R2=',R)
xax = np.linspace(-1,11)
N = -B_inv/A_inv
print('N = ',N)
N_halvsykel = 0.251189
Static_strength = A_inv*np.log10(N_halvsykel)+B_inv

print('static strength = ', Static_strength)
y_mean = sum(Y2)/float(len(Y2))
SStot = sum( [(x - y_mean)**2 for x in Y2])
SSres = sum( [(y - (A*x + B))**2 for x, y in zip(X2, Y2) ])
R2 = 1 - SSres/SStot
#print('R2 =',R2
N = -B_inv/A_inv
print('N = ',N)
N_halvsykel = 0.251189
Static_strength = A_inv*np.log10(N_halvsykel)+B_inv

print('static strength = ', Static_strength)



s = ((len(Y2) - 2)) ** (-1) * SSres
std = sqrt(s)
#print('s^2 =',s
print('s =', std)
plt.plot(Y2, X2, 'r.', label = 'Measurements T15')
plt.plot(lin[0]*xax + lin[1], xax, color='red',label = r'Regression curve T15')


X2 = data2.iloc[2: 23,53].values  # values converts it into a numpy array
Y2 = data2.iloc[2: 23,55].values  # -1 means that calculate the dimension of rows, but have 1 column

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
print('A=',A_inv, 'B=',B_inv, 'R2=',R)
y_mean = sum(Y2)/float(len(Y2))
SStot = sum( [(x - y_mean)**2 for x in Y2])
SSres = sum( [(y - (A*x + B))**2 for x, y in zip(X2, Y2) ])
R2 = 1 - SSres/SStot
#print('R2 =',R2
N = -B_inv/A_inv
print('N = ',N)
N_halvsykel = 0.251189
Static_strength = A_inv*np.log10(N_halvsykel)+B_inv

print('static strength = ', Static_strength)



s = ((len(Y2) - 2)) ** (-1) * SSres
std = sqrt(s)
#print('s^2 =',s
print('s =', std)

print('static strength = ', Static_strength)

plt.plot(Y2, X2, 'b.', label = 'Measurements T22')
plt.plot(lin2[0]*xax + lin2[1], xax, color='blue',label = 'Regression curve T22')

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
plt.title('Spruce: \n T22 vs T15')
plt.legend()
plt.show()