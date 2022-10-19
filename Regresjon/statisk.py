'''
Created on 18. feb. 2021

@author: Martinskole
'''
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools

import pandas as pd

data = pd.read_csv(r'D:\.master\Master\statiske tester\Sheartest_Force-Displacement\First12Specimens\Shear_2_A8-59_Test01_Last10Cycle+FinalLoading.csv', index_col=False )
data1 = pd.read_csv(r'D:\.master\Master\statiske tester\Sheartest_Force-Displacement\First12Specimens\Shear_2_A8-59_Test01_Last10Cycle+FinalLoading.csv')

#print(data
#print(data1
# t = time
# d = displacement
# f = force

t = data.iloc[6283:-1,0].values  # values converts it into a numpy array
d = data.iloc[6283:-1,1].values  # -1 means that calculate the dimension of rows, but have 1 column
f = data1.iloc[6283:-1,1].values

t1 = data.iloc[2:,0].values  # values converts it into a numpy array
d1 = data.iloc[2:,1].values  # -1 means that calculate the dimension of rows, but have 1 column
f1 = data1.iloc[2:,1].values


t = np.array(t)
d = np.array(d)
f = np.array(f)

t = t.astype(np.float)
d = d.astype(np.float)
f = f.astype(np.float)
t1 = np.array(t1)
d1 = np.array(d1)
f1 = np.array(f1)

t1 = t1.astype(np.float)
d1 = d1.astype(np.float)
f1 = f1.astype(np.float)

plt.plot(t, f)

plt.title('Time - Force')
plt.ylabel('[kN]')
plt.xlabel('[s]')
plt.legend()
#plt.savefig("Test13_F-t.pdf") #################################
plt.show()


X2 = f1[7620:10769] #9893
Y2 = d1[7620:10769] #10769
lin = linregress((X2, Y2))
A = lin[0]
B = lin[1]

#print('A=',A, 'b=',B, 'R2=',lin[2]**2
print('k=',1/lin[0], 'B=',lin[1]/-lin[0], 'R2=',lin[2]**2)
y_mean = sum(Y2)/float(len(Y2))
SStot = sum( [(x - y_mean)**2 for x in Y2])
SSres = sum( [(y - (A*x + B))**2 for x, y in zip(X2, Y2) ])
R2 = 1 - SSres/SStot
#print('R2 =',R2

s = ((len(Y2)-1))**(-1) * SSres
std = sqrt(s)
#print('s^2 =',s
print('s =', std)
#print('2s =', std*2

xax = np.linspace(5.2,21.1)


plt.plot(lin[0]*xax + lin[1],xax, color='red',label = 'Regressed curve')
plt.plot(Y2,X2)
plt.plot(d1,f1)
plt.title('Force - Displacement')
plt.ylabel('[kN]')
plt.xlabel('[mm]')
#plt.ylim((5, 29)) 
#plt.xlim((0.1,0.6))
plt.legend()
#plt.savefig("Test13_F-u.pdf") ###################################




f_max = np.max(f)
print('f_max =', f_max)
index =  np.argmax(f)
print('d_max =', d[index])

k = (1/lin[0])
print(k)

E = (f_max**2)/(2*k)
print(E)
E1 = ((d[index])**2)*k/(2)
print(E1)
E2 = f_max*(d[index])/2
print(E2)
######################




T = []
D = []
F = []

a = int(t[-1])+1

tot = list(range(0,a))
tot = np.array(tot)
#def isclose(a, b, rel_tol=0.04, abs_tol=0.0):
  #  return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


#for i in t:
 #   if i isclose()



plt.show()










