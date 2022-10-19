
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt
from sympy import symbols, solve

start = 59 #59 #91
slutt = 119 #119 #90

data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Energy')  # load data set
Y2 = data.iloc[start: slutt, 22].values  # values converts it into a numpy array
X2 = data.iloc[start: slutt, 21].values  # -1 means that calculate the dimension of rows, but have 1 column #lastnivå

#mod 2
# noc = data.iloc[start: slutt, 8].values
# delta_k = data.iloc[start: slutt, 9].values
# delta_k_cycle = data.iloc[start: slutt, 11].values
# E_tot_mod2 = data.iloc[start: slutt, 12].values
# E_cycle_mod2 = data.iloc[start: slutt, 13].values
# ksi_mod2 = data.iloc[start: slutt, 14].values

#total
# E_tot = data.iloc[start: slutt, 18].values
# E_cycle = data.iloc[start: slutt, 17].values
# ksi = data.iloc[start: slutt, 20].values
X3 = data.iloc[start: slutt, 22].values

#Y2 = data.iloc[start: slutt, 29].values
#X2 = data.iloc[start: slutt, 17].values

#Y2 = data.iloc[start: slutt, 34].values  # values converts it into a numpy array
#X2 = data.iloc[start: slutt, 35].values

X2 = np.array(X2)
Y2 = np.array(Y2)
X2 = X2.astype(float)
Y2 = Y2.astype(float)

#Y2 = np.log10(Y2)
X2 = np.log10(X2)

X3 = np.array(X3)

X3 = X3.astype(float)

print(X2,Y2)

lin = linregress((X2, Y2))

A = lin[0]
B = lin[1]
R = lin[2]**2

A_inv = lin[0]**-1
B_inv = lin[1]*(-lin[0])**-1


#print('A=',A, 'b=',B, 'R2=',R
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

#print('static strength = ', Static_strength)



s = ((len(Y2) - 2)) ** (-1) * SSres
std = sqrt(s)
#print('s^2 =',s
print('s =', std)

xax = np.linspace(-3,80)


plt.plot(Y2, X2, 'b.', label = 'Measurements')
plt.plot(lin[0]*xax + lin[1], xax, color='blue',label = 'Regression curve')

#plt.plot(-X3,X2, 'r.', label = 'Measurements mirrored S-N curve')
#plt.plot( (1/0.10280514048493182)*xax - 8.401604531, xax, color = 'red', label ='Mirrored S-N curve')

#plt.plot(0.6,1,'bx',label = r'$f_{u}$ spruce')
plt.ylim(-1,0)
#plt.ylim((-1.5, 0.5))  #0, 1.1 # 0, 1.3
plt.xlim((0, 10))
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)') #Damping ratio $ \xi $
plt.xlabel(r'Percent change in stiffness before failure') #Normalized stress ($f_{a}/f_{u}$)
plt.title('Pine: \n Percent change in stiffness before failure - Normalized stress')


plt.legend()
plt.show()
