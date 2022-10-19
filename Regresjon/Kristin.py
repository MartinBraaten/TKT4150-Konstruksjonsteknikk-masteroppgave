
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
#from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt

data = pd.read_excel(r'C:\Users\Martinskole\tkt4140-master2\ws\tkt4140my\src\prosjektoppgave\malinger.xls', sheet_name='Kristin') # load data set

Y2 = data.iloc[0: 15,-3].values  # values converts it into a numpy array
X2 = data.iloc[0: 15,0].values  # -1 means that calculate the dimension of rows, but have 1 column

X2 = np.array(X2)
Y2 = np.array(Y2)

lin = linregress((X2, Y2))

A = lin[0]
B = lin[1]

print('A=',A, 'b=',B, 'R2=',lin[2]**2
print('1/A=',1/lin[0], '1/B=',lin[1]/-lin[0], 'R2=',lin[2]**2
y_mean = sum(Y2)/(len(Y2))
SStot = sum( [(x - y_mean)**2 for x in Y2])
SSres = sum( [(y - (A*x + B))**2 for x, y in zip(X2, Y2) ])
R2 = 1 - SSres/SStot
print('R2 =',R2



xax = np.linspace(-1,11)

plt.plot(X2, Y2, 'b.', label = 'Data points')
plt.plot(xax,lin[0]*xax + lin[1], color='red',label = 'Regressed curve')
plt.plot(1,-0.6,'bx',label = 'Static strength')
plt.xlim((0, 1.1))   
plt.ylim((-1, 10))  
plt.ylabel('Log N')
plt.xlabel('Normalized load')
plt.title('Log N curve without static strength')
plt.legend()
plt.show()

plt.plot(Y2, X2, 'b.', label = 'Data points')
plt.plot(lin[0]*xax + lin[1], xax, color='red',label = 'Regressed curve')
plt.plot(-0.6,1,'bx',label = 'Static strength')
plt.ylim((0, 1.1))   
plt.xlim((-1, 10))
plt.xlabel('Log N')
plt.ylabel('Normalized load')
plt.title('SN curve without static strength')
plt.legend()
plt.show()


####################################################################

print('Med statisk styrke:'

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
A = lin2[0]
B = lin2[1]
print('A=',A, 'B=',B, 'R2=',lin2[2]**2

intercept = model(0, best_fit[0], x[-1], y[-1]) # The y-intercept
plt.plot(x[0:-1], y[0:-1], 'b.',label = 'Data points') # Input data will be green stars
plt.plot(1,-0.6,'bx',label = 'Static strength')
plt.plot(xax, lin2[0]*xax + lin2[1], "r-",label = 'Regressed curve') # Fit will be a red line
plt.xlim((0, 1.1))
plt.ylim((-1, 10))
plt.ylabel('Log N')
plt.xlabel('Normalized load')
plt.title('Log N curve with static strength')
plt.legend()

plt.show()

print('1/A=',1/lin2[0], '1/B=',lin2[1]/-lin2[0], 'R2=',lin2[2]**2

Y2 = np.delete(Y2,-1)
X2 = np.delete(X2,-1)

y_mean = sum(Y2)/(len(Y2))
SStot = sum( [(p - y_mean)**2 for p in Y2])
SSres = sum( [(o - (A*p + B))**2 for p, o in zip(X2, Y2) ])
R2 = 1 - SSres/SStot
print('R2 static=',R2


plt.plot(Y2, X2, 'b.', label = 'Data points')
plt.plot(lin2[0]*xax + lin2[1], xax, color='red',label = 'Regressed curve')
plt.plot(-0.6,1,'bx',label = 'Static strength')
plt.ylim((0, 1.1))   
plt.xlim((-1, 10))  
plt.xlabel('Log N')
plt.ylabel('Normalized load')
plt.title('SN curve with static strength')
plt.legend()
plt.show()