'''
Created on 21. mar. 2021

@author: Martinskole
'''
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt
from sympy import symbols, solve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


start = 1
slutt = 33
data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Furu plot')  # load data set
Y2 = data.iloc[start: slutt, 2].values  # values converts it into a numpy array
X2 = data.iloc[start: slutt, 1].values  # -1 means that calculate the dimension of rows, but have 1 column
X2 = np.array(X2)
Y2 = np.array(Y2)
X2 = X2.astype(float)
Y2 = Y2.astype(float)


#print(X2, Y2)

degree = 1
reg = np.polyfit(Y2, X2, degree, full=True) #, full=True
mymodel = np.poly1d(reg)

myline = np.linspace(-0.6, 8, 100)

plt.scatter(Y2, X2)
plt.plot(myline, mymodel(myline))
plt.show()

results = smf.ols(formula=(myline, mymodel(myline)), data=(Y2,X2)).fit()
results.summary()


'''

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


Coefficient_of_variation_Y = y_mean/s
if Coefficient_of_variation_Y < 0.05:
    print('CV er mindre enn 0.05')
else:
    print('CV_Y =',Coefficient_of_variation_Y)


xax = np.linspace(-1,11)

# 1.644853626951472714864

plt.plot(Y2, X2, 'b.', label = 'Measurements')
plt.plot(lin[0]*xax + lin[1], xax, color='red',label = 'Regression curve')

plt.plot(-0.6,1,'bx',label = r'$f_{u}$ pine')
plt.ylim((0, 1.1))  #0, 1.1 # 0, 1.3
plt.xlim((-1, 8))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Pine: \n S-N curve based on fatigue loading')

plt.plot(-0.6,0.759194286,'rx',label = r'$f_{u}$ spruce')




plt.legend()
plt.show()

res = [(y - (A*x + B)) for x, y in zip(X2, Y2)]
print(res)
plt.plot(Y2,res,'b.')
plt.plot([0,7],[0,0])
plt.xlim(0,7)
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
SStot = sum([(p - y_mean)**2 for p in Y2])
SSres = sum([(o - (A*p + B))**2 for p, o in zip(X2, Y2)])
R2 = 1 - SSres/SStot
print('R2 static=', R2)
std = sqrt(((len(Y2) - 2)) ** (-1) * SSres)
print('s =', std)
print('s*k_s(n) =', std*Char_val2)

p = symbols('p')
expr = lin2[0]*p + lin2[1]-Char_val2*std+0.6
sol = solve(expr)

print('Kar y(-0.6):',sol)
print('B char =', B_inv-(Static_strength-sol))
print('Char N =', N-std*Char_val2)
print('ss - kar=', Static_strength-sol)


plt.plot(Y2, X2, 'b.', label = 'Measurements')
plt.plot(lin2[0]*xax + lin2[1], xax, color='red',label = 'Regression curve')
plt.plot(lin2[0]*xax + lin2[1]-Char_val2*std, xax, color='green',label = 'Characteristic curve')
plt.plot(-0.6,1,'bx',label = r'$f_{u}$ Pine')
plt.ylim((0, 1.1))   # 1.6 med statisk forsok, 1.1 ellers
plt.xlim((-1, 8))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Pine: \n S-N curve based on fatigue loading and static strength')

plt.plot(-0.6,0.759194286,'rx',label = r'$f_{u}$ spruce')

plt.legend()
plt.show()
'''