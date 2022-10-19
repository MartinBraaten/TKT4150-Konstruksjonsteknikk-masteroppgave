import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # To visualize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit


start = 1
slutt = 33
data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Furu plot')  # load data set
Y2 = data.iloc[start: slutt, 2].values  # values converts it into a numpy array
X2 = data.iloc[start: slutt, 1].values  # -1 means that calculate the dimension of rows, but have 1 column
X2 = np.array(X2)
Y2 = np.array(Y2)
X2 = X2.astype(float)
Y2 = Y2.astype(float)
print(X2,Y2)
#Y2 = Y2.reshape(-1,1)
#X2 = X2.reshape(-1,1)
#x = X2
#y = Y2

#def func(x, a, b):#, c):
#    return a*x+b   #a * np.exp(-b * x) + c
def sigmoid(x,x0, k):
    y = -1 / (1 + np.exp(-k*(x-x0)))+1
    return (y)

def sigmoid1(x,x0, k, L, b):
    y2 = -L / (1 + np.exp(-k*(x-x0)))+b
    return (y2)


popt, pcov = curve_fit(sigmoid, Y2, X2)
popt1, pcov1 = curve_fit(sigmoid1, Y2, X2)
print(popt)
print(popt1)

x1 =  np.linspace(-1, 10, 32)
y1 = -1/ (1 + np.exp(-0.95100622*(x1-3.5)))+1
y = np.linspace(-5, 15, 32)
y2 = np.linspace(-5, 15, 32)

x = sigmoid(y, *popt)
x2 = sigmoid1(y2, *popt1)
plt.plot(y, y1, label='fit1') #x0=4.62325022
plt.plot(y2, x2, label='fit2')
plt.plot(-0.6,1,'bx', label='f_u')
#plt.plot(y, x, 'o', label='data')
plt.plot(y, x, label='fit3')
plt.plot(Y2,X2,'.')
plt.ylim(0, 1.1)
plt.legend()
plt.show()

res = (Y2-y)
print(res)
plt.plot(Y2,res,'b.', label = 'Residuals')
plt.plot([0,7],[0,0])
plt.xlim(1,7)
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel('Residuals')
plt.title('Pine: \n Residual plot')
plt.legend()
plt.show()



#plt.plot(y, x, 'b.', label='data')
#plt.show()
#popt, pcov = curve_fit(func, x, y)

#print("a =", 1/popt[0], "+/-", pcov[0,0]**0.5)
#print("b =", -popt[1]/popt[0], "+/-", pcov[1,1]**0.5)
#xfine = np.linspace(-1., 10., 100)  # define values to plot the function for
#plt.plot(func(xfine, popt[0], popt[1]), xfine, 'r-')
#plt.plot(y, x, 'b.', label='data')
#plt.xlim(-1,7)
#plt.ylim(0,1.1)
#plt.xlabel('Log N')
#plt.ylabel('Normalized load')
#plt.legend()
#plt.show()