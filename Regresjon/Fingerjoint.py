import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from scipy.stats import linregress
from scipy.optimize import curve_fit
import functools
from math import sqrt
from sympy import symbols, solve

data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='R=-1')  # load data set


Y2 = data.iloc[1: 27, 20].values  # values converts it into a numpy array
X2 = data.iloc[1: 27, 19].values  # -1 means that calculate the dimension of rows, but have 1 column
Y1 = data.iloc[1: 4, 18].values  # values converts it into a numpy array
X1 = data.iloc[1: 4, 17].values  # -1 means that calculate the dimension of rows, but have 1 column

X1 = np.array(X1)
X2 = np.array(X2)
Y2 = np.array(np.log10(Y2))
Y1 = np.array(np.log10(Y1))

print('len y =', len(Y2))
Char_val2 = (6.5 * len(Y2) + 6) / (3.7 * len(Y2) - 3)
print('Char_val2 =', Char_val2)

print('len y =', len(Y1))
Char_val1 = (6.5 * len(Y1) + 6) / (3.7 * len(Y1) - 3)
print('Char_val1 =', Char_val1)

lin = linregress((X2, Y2))
lin2 = linregress((X1, Y1))

A = lin[0]
B = lin[1]
R = lin[2] ** 2

A2 = lin2[0]
B2 = lin2[1]
R1 = lin2[2] ** 2



A_inv = lin[0] ** -1
B_inv = lin[1] * (-lin[0]) ** -1


print('A=', A_inv, 'B=', B_inv, 'R2=', R)
y_mean = sum(Y2) / float(len(Y2))
SStot = sum([(x - y_mean) ** 2 for x in Y2])
SSres = sum([(y - (A * x + B)) ** 2 for x, y in zip(X2, Y2)])
R2 = 1 - SSres / SStot
# print('R2 =',R2
N = -B_inv / A_inv
print('N = ', N)
N_halvsykel = 0.251189
Static_strength = A_inv * np.log10(N_halvsykel) + B_inv

print('static strength = ', Static_strength)

s = ((len(Y2) - 2)) ** (-1) * SSres
std = sqrt(s)
# print('s^2 =',s
print('s =', std)
print('s*char_val =', std * Char_val2)



A_inv2 = lin2[0] ** -1
B_inv2 = lin2[1] * (-lin2[0]) ** -1
print()
print(lin2[3],lin2[4])
print(lin[3],lin[4]*sqrt((len(Y2)-1)/(len(Y2)-2)))

print('A1=', A_inv2, 'B1=', B_inv2, 'R21=', R1)
y_mean = sum(Y1) / float(len(Y1))
SStot = sum([(x - y_mean) ** 2 for x in Y1])
SSres = sum([(y - (A * x + B)) ** 2 for x, y in zip(X1, Y1)])
R2 = 1 - SSres / SStot
# print('R2 =',R2
N1 = -B_inv2 / A_inv2
print('N2 = ', N1)

Static_strength2 = A_inv2 * np.log10(N_halvsykel) + B_inv2

print('static strength = ', Static_strength2)

s1 = ((len(Y1) - 2)) ** (-1) * SSres
std1 = sqrt(s1)
# print('s^2 =',s
print('s^2 =,',s1)
print('s1 =', std1)
print('s1*char_val1 =', std1 * Char_val1)


p = symbols('p')
expr = lin[0] * p + lin[1] - Char_val2 * std + 0.6
sol = solve(expr)

print('ss - kar=', Static_strength - sol)
print('Kar verdi i -0.6:', sol)
print('N char =', N - std * Char_val2)

p = symbols('p')
expr1 = lin[0] * p + lin[1] - Char_val1 * std1 + 0.6
sol1 = solve(expr1)

print('ss - kar=', Static_strength2 - sol1)
print('Kar verdi i -0.6:', sol1)
print('N char =', N1 - std1 * Char_val1)






xax = np.linspace(-1, 11)

plt.plot(Y1, X1, 'r.', label='Measurements with finger joint')
plt.plot(Y2, X2, 'b.', label='Measurements without finger joint')

plt.plot(lin2[0] * xax + lin2[1], xax, color='red', label='Regression curve with finger joint')
plt.plot(lin[0] * xax + lin[1], xax, color='blue', label='Regression curve without finger joint')

plt.plot(-0.6,1,'bx',label = r'$f_{u}$ spruce')
plt.ylim((0, 1.2))  # 0, 1.1 # 0, 1.6
plt.xlim((-1, 10))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Spruce: \n Finger joint vs without finger joint')



plt.legend()
plt.show()

