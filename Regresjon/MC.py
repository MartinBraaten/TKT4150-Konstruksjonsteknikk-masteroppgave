
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
#from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt
from sympy import symbols, solve

data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='MC') # load data set

start = 3
slutt = 30
Y2 = data.iloc[start: slutt,1].values  # values converts it into a numpy array
X2 = data.iloc[start: slutt,0].values  # Density
X3 = data.iloc[start: slutt,1].values  # MC
X4 = data.iloc[start: slutt,1].values  # Annual

X2 = np.array(X2)
Y2 = np.array(Y2)
X3 = np.array(X3)
X4 = np.array(X4)

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

print('A=',A_inv, 'B=',B_inv, 'R2=',R, 'p = ',lin[3], 'stderr =',lin[4])

xax = np.linspace(-1,1.2)

plt.plot(Y2, X2, 'r.', label = 'Measurements')
plt.plot(lin[0]*xax + lin[1], xax, color='red',label = 'Regression curve')
plt.ylim(-0.01,0.003)
plt.xlim((-0.03*1000000),(1.7*1000000))
plt.xlabel('Time (s)')
plt.ylabel(r'MC [%]')
plt.title('Spruce: \n Moisture content lost per second')
plt.legend()
plt.show()
###############################################
lin2 = linregress((X3, Y2))

A = lin2[0]
B = lin2[1]
R = lin2[2]**2

A_inv = lin2[0]**-1
B_inv = lin2[1]*(-lin2[0])**-1
print('A=',A_inv, 'B=',B_inv, 'R2=',R, 'p = ',lin2[3], 'stderr =',lin2[4])

xax = np.linspace(10,13)

plt.plot(Y2, X3, 'b.', label = 'Measurements')
plt.plot(lin2[0]*xax + lin2[1], xax, color='blue',label = 'Regression curve')


#plt.plot(-0.6,1,'bx',label = r'$f_{u}$ spruce')
#plt.ylim((0, 1.4))  #0, 1.1 # 0, 1.6
#plt.xlim((-1, 11))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel('Moisture content [%]')
plt.title('Spruce: \n Moisture content')
plt.legend()
#plt.show()

