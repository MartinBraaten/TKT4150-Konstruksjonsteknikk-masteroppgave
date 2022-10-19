import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt
from sympy import symbols, solve
 
data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='R=-1') # load data set
data1 = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Statisk Gran') # load data set

X2 = data.iloc[1: 30,5].values  # values converts it into a numpy array
Y2 = data.iloc[1: 30,4].values  # -1 means that calculate the dimension of rows, but have 1 column
X2 = np.array(X2)
Y2 = np.array(Y2)
X2 = X2.astype(float)
Y2 = Y2.astype(float)

print(X2,Y2)
print('len y =', len(Y2))
Char_val2 = (6.5*len(Y2)+6)/(3.7*len(Y2)-3)
print('k_s(n) gran syklisk =',Char_val2)
#print(data,X2,Y2)
#'''
# K er Statiske forsok for gran fra Kvittingen
K = data1.iloc[0: 51,8].values
K = np.array(K)
L = [-0.6]*len(K)

y_mean1 = sum(K)/float(len(K))
SStot1 = sum( [(x - y_mean1)**2 for x in K])
SSres1 = sum( [(y - 1)**2 for x, y in zip(L, K) ])
R21 = 1 - SSres1/SStot1
s3 = sqrt(((len(K)-1))**(-1) * SSres1)
s2 = 0.05*y_mean1
s1 = max(s2,s3)
print('vanlig =',s3, '0.05*mean =',s2)
print('s1 =',s1)

#print(X2, Y2)

print('len K =', len(K))
Char_val_K = (6.5*len(K)+6)/(3.7*len(K)-3)
print('k_s(n)_K =',Char_val_K)

print('Standard avvik statisk forsok:',s1)
Kar = 1-(s1*Char_val_K)
print('Karakteristisk verdi statisk forsok:',Kar)

plt.plot(L, K,'m.', label = 'Measurements from Kvittingen')
plt.plot(-0.6,Kar,'gx',label = 'Characteristic value')

Coefficient_of_variation_K = y_mean1/s1
if Coefficient_of_variation_K < 0.05:
    print('CV er mindre enn 0.05')
else:
    print('CV_K =',Coefficient_of_variation_K)
#'''

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


SSres4 = sum([(x - (y-B)/A)**2 for x, y in zip(X2, Y2)])
s4 = ((len(Y2) - 2)) ** (-1) * SSres4
std4 = sqrt(s4)
print('s i y-retning=', std4)


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

fatigue = [N,-0.6]
karakt = [0,Kar,0,Kar]
fatigue1 = [N-std*Char_val2,-0.6]


plt.plot(Y2, X2, 'b.', label = 'Measurements')
plt.plot(7,0.25,'+',color='black')
plt.plot(7,0.3,'+',color='black')
plt.plot(7.191987877,0.2,'+',color='black', label = 'Did not fail')
plt.plot(-0.6,1,'bx',label = r'$f_{u}$ spruce')
plt.plot(lin[0]*xax + lin[1], xax, color='red',label = 'Regression curve')
plt.plot(lin[0]*xax + lin[1]-Char_val2*std,xax, color='green',label = 'Characteristic curve') #1.644854
for i in range(0, len(fatigue), 2):
    plt.plot(fatigue[i:i+2], karakt[i:i+2],color='black', label = 'Relative decline in characteristic value')
for i in range(0, len(fatigue), 2):
    plt.plot(fatigue1[i:i+2], karakt[i:i+2],color='blue', label = 'Most conservative for design')

plt.ylim((0, 1.6))  #0, 1.1 # 0, 1.6
plt.xlim((-1, 10))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Spruce: \n S-N curve based on fatigue loading')


p = symbols('p')
expr = lin[0]*p + lin[1]-Char_val2*std+0.6
sol = solve(expr)

print('ss - kar=', Static_strength-sol)
print('Kar verdi i -0.6:',sol)
print('N char =', N-std*Char_val2)

print('B char =', B_inv-(Static_strength-sol))
print('B char N =', N-std*Char_val2)


#plt.plot(-0.6,sol, 'rx', label = 'kar val')

'''

a = 6.7 #8.2 # org 6.7
b = 1.3
beta = 1 #0.37 # org 1
t = 100
R = -1
beta_3 = 3

#stigning = a
l = symbols('l')
expr3 = (1-R)/(l*(b-R)) + A_inv
#a = float(solve(expr3)[0])
print(a)
N = np.linspace(0,1, 10000)
N_1 = np.linspace(1,50000000)
k_fat = 1 - (1-R)/(a*(b-R))*np.log10(beta*N[1:-1]*t)
k_fat_1 = 1 - (1-R)/(a*(b-R))*np.log10(beta*N_1*t)

plt.plot(np.log10(N[1:-1]), k_fat,color='y',label =  r'$ k_{fat}, \beta $ = 1')
plt.plot(np.log10(N_1), k_fat_1,color='y')
k_fat_3 = 1 - (1-R)/(a*(b-R))*np.log10(beta_3*N[1:-1]*t)
k_fat_1_3 = 1 - (1-R)/(a*(b-R))*np.log10(beta_3*N_1*t)
# 
# 
plt.plot(np.log10(N[1:-1]), k_fat_3,color='black',label =  r'$ k_{fat}, \beta $ = 3') #np.log10(
plt.plot(np.log10(N_1), k_fat_1_3,color='black')
'''

plt.legend()
plt.show()

res4= [(x - (y-B)/A) for x, y in zip(X2, Y2)]

res = [(y - (A*x + B)) for x, y in zip(X2, Y2)]

plt.plot(X2,res,'b.', label = 'Residuals')


plt.plot([0,0.7],[0,0])
plt.xlim(0.1,0.7)
plt.xlabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.ylabel('Residuals')
plt.title('Spruce: \n Residual plot')
plt.legend()
plt.show()
'''
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
plt.plot(7,0.25,'+',color='black')
plt.plot(7,0.3,'+',color='black')
plt.plot(7.191987877,0.2,'+',color='black', label = 'Did not fail')
plt.plot(lin2[0]*xax + lin2[1], xax, color='red',label = 'Regression curve')
plt.plot(lin2[0]*xax + lin2[1]-Char_val2*std, xax, color='green',label = 'Characteristic curve')
plt.plot(-0.6,1,'bx',label = r'$f_{u}$ spruce')
plt.ylim((0, 1.6))   # 1.6 med statisk forsok, 1.1 ellers
plt.xlim((-1, 10))  
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Spruce: \n S-N curve based on fatigue loading and static strength')

#'''
plt.plot(L, K,'m.', label = 'Measurements from Kvittingen')
plt.plot(-0.6,Kar,'gx',label = 'Characteristic value')
#'''
'''
a = 6.7 #8.2
b = 1.3
beta = 1 #0.3
t = 100
R = -1
beta_3 = 3

N = np.linspace(0,1, 10000)
N_1 = np.linspace(1,500000)
k_fat = 1 - (1-R)/(a*(b-R))*np.log10(beta*N[1:-1]*t)
k_fat_1 = 1 - (1-R)/(a*(b-R))*np.log10(beta*N_1*t)

plt.plot(np.log10(N[1:-1]), k_fat,color='y',label =  r'$ k_{fat}, \beta $ = 1')
plt.plot(np.log10(N_1), k_fat_1,color='y')
k_fat_3 = 1 - (1-R)/(a*(b-R))*np.log10(beta_3*N[1:-1]*t)
k_fat_1_3 = 1 - (1-R)/(a*(b-R))*np.log10(beta_3*N_1*t)

plt.plot(np.log10(N[1:-1]), k_fat_3,color='black',label =  r'$ k_{fat}, \beta $ = 3') #np.log10(
plt.plot(np.log10(N_1), k_fat_1_3,color='black')
'''
plt.legend()
#plt.show()
