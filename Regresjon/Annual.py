
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
#from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt
from sympy import symbols, solve

data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='R=-1') # load data set

start = 33
slutt = 62
Y2 = data.iloc[start: slutt,0].values  # values converts it into a numpy array
X2 = data.iloc[start: slutt,1].values  # Density
X3 = data.iloc[start: slutt,2].values  # MC
X4 = data.iloc[start: slutt,3].values  # Annual

X2 = np.array(X2)
Y2 = np.array(Y2)
X3 = np.array(X3)
X4 = np.array(X4)

X2 = X2.astype(float)
Y2 = Y2.astype(float)
X3 = X3.astype(float)
X4 = X4.astype(float)


print('len y =', len(Y2))


lin = linregress((X2, Y2))

A = lin[0]
B = lin[1]
R = lin[2]**2

A_inv = lin[0]**-1
B_inv = lin[1]*(-lin[0])**-1

print('A=',A_inv, 'B=',B_inv, 'R2=',R, 'p = ',lin[3], 'stderr =',lin[4])

xax = np.linspace(350,500)

plt.plot(Y2, X2, 'r.', label = 'Measurements')
plt.plot(lin[0]*xax + lin[1], xax, color='red',label = 'Regression curve')

plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Density $[kg/m^{3}]$')
plt.title('Spruce: \n Density')
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


plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel('Moisture content [%]')
plt.title('Spruce: \n Moisture content')
plt.legend()
plt.show()


######################################

lin3 = linregress((X4, Y2))

A = lin3[0]
B = lin3[1]
R = lin3[2]**2

A_inv = lin3[0]**-1
B_inv = lin3[1]*(-lin3[0])**-1
print('A=',A_inv, 'B=',B_inv, 'R2=',R, 'p = ',lin3[3], 'stderr =',lin3[4])

xax = np.linspace(10,110)

plt.plot(Y2, X4, 'b.', label = 'Measurements')
plt.plot(lin3[0]*xax + lin3[1], xax, color='blue',label = 'Regression curve')



#plt.plot(-0.6,1,'bx',label = r'$f_{u}$ spruce')
#plt.ylim((0, 1.4))  #0, 1.1 # 0, 1.6
#plt.xlim((-1, 11))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel('Annual rings')
plt.title('Spruce: \n Annual rings')
plt.legend()
plt.show()