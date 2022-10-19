
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt
from sympy import symbols, solve

#spruce
start = 90 #59 #90
slutt = 119 #119 #90
data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Energy')  # load data set
Y = data.iloc[start: slutt, 22].values
X = data.iloc[start: slutt, 21].values
#pine
X5 = data.iloc[59: 90, 21].values
Y5 = data.iloc[59: 90, 22].values
X5 = np.array(X5*1.317185889)
X5 = X5.astype(float)
Y5 = np.array(Y5)
Y5 = Y5.astype(float)

X = np.array(X)
Y = np.array(Y)
X = X.astype(float)
Y = Y.astype(float)

print(X,Y)
print(X5,Y5)

X2 = []
Y2 = []
X3 = []
Y3 = []
for i in range(len(Y)):
    if Y[i] < 4:
        Y2.append(Y[i])
        X2.append(X[i])
    else:
        Y3.append(Y[i])
        X3.append(X[i])

for i in range(len(Y5)):
    if Y5[i] < 4:
        Y2.append(Y5[i])
        X2.append(X5[i])
    else:
        Y3.append(Y5[i])
        X3.append(X5[i])

print(X2, Y2)
print(X3, Y3)


############# <0.4
print('len y =', len(Y2))
Char_val2 = (6.5*len(Y2)+6)/(3.7*len(Y2)-3)
print('k_s(n) furu syklisk =',Char_val2)

lin = linregress((X2, Y2))

A = lin[0]
B = lin[1]
R = lin[2]**2

A_inv = lin[0]**-1
B_inv = lin[1]*(-lin[0])**-1

print('A=',A_inv, 'B=',B_inv, 'R2=',R)
y_mean = sum(Y2)/float(len(Y2))
SStot = sum( [(x - y_mean)**2 for x in Y2])
SSres = sum( [(y - (A*x + B))**2 for x, y in zip(X2, Y2) ])
R2 = 1 - SSres/SStot

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

Coefficient_of_variation_Y = y_mean/s
if Coefficient_of_variation_Y < 0.05:
    print('CV er mindre enn 0.05')
else:
    print('CV_Y =',Coefficient_of_variation_Y)


xax = np.linspace(-1,11)

plt.plot(Y2, X2, 'b.', label = 'Measurements')
#plt.plot(7,0.16,'+',color='black', label = 'Did not fail')
plt.plot(lin[0]*xax + lin[1], xax, color='red',label = 'Regression curve')
plt.plot(lin[0]*xax + lin[1]-Char_val2*std,xax, color='green',label = 'Characteristic curve') #1.644854
plt.plot(-0.6,1,'bx',label = r'$f_{u}$ ')
plt.ylim((0, 1.1))  #0, 1.1 # 0, 1.3
plt.xlim((-1, 8))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Combined: \n S-N curve based on fatigue loading')

#plt.plot(-0.6,0.759194286,'rx',label = r'$f_{u}$ spruce')



p = symbols('p')
expr = lin[0]*p + lin[1]-Char_val2*std+0.6
sol = solve(expr)

print('ss - kar=', Static_strength-sol)
print('Kar verdi i -0.6:',sol)
print('N char =', N-std*Char_val2)

print('B char =', B_inv-(Static_strength-sol))
print('B char N =', N-std*Char_val2)


plt.legend()
plt.show()

res = [(y - (A*x + B)) for x, y in zip(X2, Y2)]

plt.plot(X2,res,'b.', label = 'Residuals')
plt.plot([0,0.9],[0,0])
plt.xlim(0.1,0.7)
plt.ylim(-1.3,1)
plt.xlabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.ylabel('Residuals')
plt.title('Combined: \n Residual plot for N$<10^4$')
plt.legend()
plt.show()

##################### >0.4

print('len y =', len(Y3))
Char_val2 = (6.5*len(Y3)+6)/(3.7*len(Y3)-3)
print('k_s(n) furu syklisk =',Char_val2)

lin1 = linregress((X3, Y3))

A = lin1[0]
B = lin1[1]
R = lin1[2]**2

A_inv = lin1[0]**-1
B_inv = lin1[1]*(-lin1[0])**-1

print('A=',A_inv, 'B=',B_inv, 'R2=',R)
y_mean = sum(Y3)/float(len(Y3))
SStot = sum( [(x - y_mean)**2 for x in Y3])
SSres = sum( [(y - (A*x + B))**2 for x, y in zip(X3, Y3) ])
R2 = 1 - SSres/SStot

N = -B_inv/A_inv
print('N = ',N)
N_halvsykel = 0.251189
Static_strength = A_inv*np.log10(N_halvsykel)+B_inv

print('static strength = ', Static_strength)

s = ((len(Y3) - 2)) ** (-1) * SSres
std = sqrt(s)
#print('s^2 =',s
print('s =', std)
print('s*char_val =', std*Char_val2)

Coefficient_of_variation_Y = y_mean/s
if Coefficient_of_variation_Y < 0.05:
    print('CV er mindre enn 0.05')
else:
    print('CV_Y =',Coefficient_of_variation_Y)


xax = np.linspace(-1,11)

plt.plot(Y3, X3, 'b.', label = 'Measurements')
plt.plot(7,0.16,'+',color='black', label = 'Did not fail')
plt.plot(lin1[0]*xax + lin1[1], xax, color='red',label = 'Regression curve')
plt.plot(lin1[0]*xax + lin1[1]-Char_val2*std,xax, color='green',label = 'Characteristic curve') #1.644854
plt.plot(-0.6,1,'bx',label = r'$f_{u}$')
plt.ylim((0, 1.1))  #0, 1.1 # 0, 1.3
plt.xlim((-1, 8))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Combined: \n S-N curve based on fatigue loading')

#plt.plot(-0.6,0.759194286,'rx',label = r'$f_{u}$ spruce')



p = symbols('p')
expr = lin1[0]*p + lin1[1]-Char_val2*std+0.6
sol = solve(expr)

print('ss - kar=', Static_strength-sol)
print('Kar verdi i -0.6:',sol)
print('N char =', N-std*Char_val2)

print('B char =', B_inv-(Static_strength-sol))
print('B char N =', N-std*Char_val2)


plt.legend()
plt.show()

res = [(y - (A*x + B)) for x, y in zip(X3, Y3)]

plt.plot(X3,res,'b.', label = 'Residuals')


plt.plot([0,0.9],[0,0])
plt.xlim(0.1,0.7)
plt.xlabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.ylabel('Residuals')
plt.ylim(-1.3,1)
plt.title('Combined: \n Residual plot for N$>10^4$')
plt.legend()
plt.show()


xax = np.linspace(-1,11)

plt.plot(Y2, X2, 'r.', label = r'Measurements N$<10^4$') #Y2, X2,
plt.plot(Y3, X3, 'b.', label = r'Measurements N$>10^4$') #Y3, X3,
#plt.plot(7,0.16,'+',color='black', label = 'Did not fail')
plt.plot(lin[0]*xax + lin[1], xax, color='red',label = 'Regression curve N$<10^4$')
plt.plot(lin1[0]*xax + lin1[1], xax, color='blue',label = 'Regression curve N$>10^4$')
#plt.plot(lin1[0]*xax + lin1[1]-Char_val2*std,xax, color='green',label = 'Characteristic curve') #1.644854
plt.plot(-0.6,1,'bx',label = r'$f_{u,pine}$')
plt.ylim((0, 1.1))  #0, 1.1 # 0, 1.3
plt.xlim((-1, 8))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Combined: \n S-N curve based on fatigue loading')
plt.legend()
plt.show()
