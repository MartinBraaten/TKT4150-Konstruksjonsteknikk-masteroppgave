'''
Created on 15. mar. 2021

@author: Martinskole
'''

from math import sqrt
import matplotlib.pyplot as plt  # To visualize
import numpy as np
import pandas as pd  # To read data


data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Statisk Gran') # load data set

print(data)
#X = data.iloc[1: 30,-1].values  # values converts it into a numpy array
#X = np.array(X)
K = data.iloc[0: 51,8].values  # -1 means that calculate the dimension of rows, but have 1 column
K = np.array(K)
L = [-0.6]*len(K)

print('len y =', len(K))
Char_val2 = (6.5*len(K)+6)/(3.7*len(K)-3)
print('Char_val2 =',Char_val2)


y_mean1 = sum(K)/float(len(K))
SStot1 = sum( [(x - y_mean1)**2 for x in K])
SSres1 = sum( [(y - 1)**2 for x, y in zip(L, K) ])
R21 = 1 - SSres1/SStot1
s3 = ((len(K)-1))**(-1) * SSres1
s2 = 0.05*y_mean1
s1 = max(s2,s3)
print(s2,s3)
std1 = sqrt(s1)

print('std =',std1)
Kar = 1-Char_val2*std1

print(Kar)
plt.plot(L, K,'m.', label = 'Measurements from Kvittingen')
plt.plot(-0.6,Kar,'gx',label = 'Characteristic value')
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylim((0, 1.6))   
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Spruce: \n Static tests with characteristic value')
plt.legend()
plt.show()

