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


X2 = data2.iloc[1: 12,0].values  # values converts it into a numpy array
Y2 = data2.iloc[1: 12,1].values  # -1 means that calculate the dimension of rows, but have 1 column
# 0 , 4
X2 = np.array(X2)
Y2 = np.array(Y2)
X2 = X2.astype(float)
Y2 = Y2.astype(float)
print(X2,Y2)
print('len y =', len(Y2))
Char_val2 = (6.5*len(Y2)+6)/(3.7*len(Y2)-3)
print('Char_val2 =',Char_val2)
# K er Statiske forsok for gran fra Kvittingen
# K = data1.iloc[0: 51,0].values  
# K = np.array(K)
# L = [-0.6]*len(K)
#  
# y_mean1 = sum(K)/float(len(K))
# SStot1 = sum( [(x - y_mean1)**2 for x in K])
# SSres1 = sum( [(y - 1)**2 for x, y in zip(L, K) ])
# R21 = 1 - SSres1/SStot1
# s1 = ((len(K)-1))**(-1) * SSres1
# std1 = sqrt(s1)
#  
# #print(X2, Y2
#  
# print('Standard avvik statisk forsok:',std1
# Kar = 1-(std1*Char_val2)
# print('Karakteristisk verdi statisk forsok:',Kar
#  
# plt.plot(L, K,'m.', label = 'Data points')
# plt.plot(-0.6,Kar,'gx',label = 'Characteristic value')


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

print('static strength = ', Static_strength)


s = ((len(Y2) - 2)) ** (-1) * SSres
std = sqrt(s)
#print('s^2 =',s
print('s =', std)
print('s*char_val =', std*Char_val2)

xax = np.linspace(-1,11)

# 1.644853626951472714864

plt.plot(Y2, X2, 'b.', label = 'Measurements with knots excluded')
plt.plot(lin[0]*xax + lin[1], xax, color='red',label = 'Regression curve')
plt.plot(lin[0]*xax + lin[1]-Char_val2*std,xax, color='green',label = 'Characteristic curve') #1.644854
plt.plot(-0.6,1,'bx',label = r'$f_{u}$ spruce')
plt.ylim((0, 1.4))  #0, 1.1 # 0, 1.6
plt.xlim((-1, 10))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Spruce: \n S-N curve based on fatigue loading')


p = symbols('p')
expr = lin[0]*p + lin[1]-Char_val2*std+0.6
sol = solve(expr)

print('B char =', B_inv-(Static_strength-sol))
print('Kar verdi i -0.6:',sol)
print('ss - kar=', Static_strength-sol)
print('N char =', N-std*Char_val2)
#plt.plot(-0.6,sol, 'rx', label = 'kar val')
# 
# a = 6.7 #8.2
# b = 1.3
# beta = 1 #0.3
# t = 100
# R = -1
# beta_3 = 3
#  
# N = np.linspace(0,1, 10000)
# N_1 = np.linspace(1,500000)
# k_fat = 1 - (1-R)/(a*(b-R))*np.log10(beta*N[1:-1]*t)
# k_fat_1 = 1 - (1-R)/(a*(b-R))*np.log10(beta*N_1*t)
#  
# plt.plot(np.log10(N[1:-1]), k_fat,color='y',label =  r'$ k_{fat}, \beta $ = 1')
# plt.plot(np.log10(N_1), k_fat_1,color='y')
# k_fat_3 = 1 - (1-R)/(a*(b-R))*np.log10(beta_3*N[1:-1]*t)
# k_fat_1_3 = 1 - (1-R)/(a*(b-R))*np.log10(beta_3*N_1*t)
# 
# 
# plt.plot(np.log10(N[1:-1]), k_fat_3,color='black',label =  r'$ k_{fat}, \beta $ = 3') #np.log10(
# plt.plot(np.log10(N_1), k_fat_1_3,color='black')
plt.legend()
plt.show()


####################################################################
print('')
print('')
print('Med statisk styrke:')

def model(x, m, xf, yf):
    return m*(x-xf)+yf

X2 = np.append(X2,[1])
Y2 = np.append(Y2,[-0.6])
x = X2
y = Y2

partial_model = functools.partial(model, xf=x[-1], yf=y[-1])
p0 = [y[-1]/x[-1]] # Initial guess for m, as long as xf != 0

best_fit, covar = curve_fit(partial_model, x, y, p0=p0)
y_fit = model(x, best_fit[0], x[-1], y[-1])
lin2 = linregress(x,y_fit)

intercept = model(0, best_fit[0], x[-1], y[-1]) 


A = lin2[0]
B = lin2[1]
R = lin2[2]**2

A_inv = lin2[0]**-1
B_inv = lin2[1]*(-lin2[0])**-1
print('A=',A_inv, 'B=',B_inv, 'R2=',R)

N = -B_inv/A_inv
print('N = ',N)
Static_strength = A_inv*np.log10(N_halvsykel)+B_inv
print('static strength = ', Static_strength,'= 1')

Y2 = np.delete(Y2,-1)
X2 = np.delete(X2,-1)

y_mean = sum(Y2)/(len(Y2))
SStot = sum( [(p - y_mean)**2 for p in Y2])
SSres = sum( [(o - (A*p + B))**2 for p, o in zip(X2, Y2) ])
R2 = 1 - SSres/SStot
print('R2 static=',R2)
s = ((len(Y2) - 2)) ** (-1) * SSres
std = sqrt(s)
#print('s^2 =',s
print('s =', std)
print('s*char_val =', std*Char_val2)

p = symbols('p')
expr = lin2[0]*p + lin2[1]-Char_val2*std+0.6
sol = solve(expr)

print('Kar verdi i -0.6:',sol)
print('ss - kar=', Static_strength-sol)


print('B char =', B_inv-(Static_strength-sol))
print('B char N =', N-std*Char_val2)

plt.plot(Y2, X2, 'b.', label = 'Measurements with knots excluded')
plt.plot(lin2[0]*xax + lin2[1], xax, color='red',label = 'Regression curve')
plt.plot(lin2[0]*xax + lin2[1]-Char_val2*std, xax, color='green',label = 'Characteristic curve')
plt.plot(-0.6,1,'bx',label = r'$f_{u}$ spruce')
plt.ylim((0, 1.1))   
plt.xlim((-1, 10))  
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Spruce: \n S-N curve based on fatigue loading and static strength')


# a = 6.7 #8.2
# b = 1.3
# beta = 1 #0.3
# t = 100
# R = -1
# beta_3 = 3
#  
# N = np.linspace(0,1, 10000)
# N_1 = np.linspace(1,500000)
# k_fat = 1 - (1-R)/(a*(b-R))*np.log10(beta*N[1:-1]*t)
# k_fat_1 = 1 - (1-R)/(a*(b-R))*np.log10(beta*N_1*t)
#  
# plt.plot(np.log10(N[1:-1]), k_fat,color='y',label =  r'$ k_{fat}, \beta $ = 1')
# plt.plot(np.log10(N_1), k_fat_1,color='y')
# k_fat_3 = 1 - (1-R)/(a*(b-R))*np.log10(beta_3*N[1:-1]*t)
# k_fat_1_3 = 1 - (1-R)/(a*(b-R))*np.log10(beta_3*N_1*t)
# 
# 
# plt.plot(np.log10(N[1:-1]), k_fat_3,color='black',label =  r'$ k_{fat}, \beta $ = 3') #np.log10(
# plt.plot(np.log10(N_1), k_fat_1_3,color='black')
plt.legend()
plt.show()