'''
Created on 14. apr. 2021

@author: Martinskole
'''

from math import sqrt
import matplotlib.pyplot as plt  # To visualize
import numpy as np
import pandas as pd  # To read data


data1 = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Statisk Gran') # load data set

data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Statisk furu') # load data set


#X = data.iloc[1: 30,-1].values  # values converts it into a numpy array
#X = np.array(X)
K = data.iloc[1: 28, 1].values  # -1 means that calculate the dimension of rows, but have 1 column
K = np.array(K)
L = [-0.6]*len(K)
print(K)

print('len K =', len(K))
Char_val2 = (6.5*len(K)+6)/(3.7*len(K)-3)
print('Char_val2 =',Char_val2)


y_mean1 = sum(K)/float(len(K))
SStot1 = sum( [(x - y_mean1)**2 for x in K])
SSres1 = sum( [(y - y_mean1)**2 for x, y in zip(L, K) ])
R21 = 1 - SSres1/SStot1
s1 = (len(K) - 1) ** (-1) * SSres1
std1 = sqrt(s1)
print(y_mean1)
print('std =',std1)
Kar = y_mean1-Char_val2*std1 #*1.317185889
print('Kar_val = ',Kar)

plt.plot(L, K,'b.', label = 'Measurements pine')
plt.plot(-0.6,Kar,'gx',label = 'Characteristic value pine')


#################################################################
K1 = data1.iloc[0: 51,8].values  # -1 means that calculate the dimension of rows, but have 1 column
K1 = np.array(K1) #/1.317185889)
L1 = [-0]*len(K1)

print('len y =', len(K1))
Char_val1 = (6.5*len(K1)+6)/(3.7*len(K1)-3)
print('Char_val2 =',Char_val1)


y_mean1 = sum(K1)/float(len(K1))
SStot11 = sum( [(x - y_mean1)**2 for x in K1])
SSres11 = sum( [(y - 1)**2 for x, y in zip(L1, K1) ])
R211 = 1 - SSres11/SStot11
s11 = (len(K1) - 1) ** (-1) * SSres11
std11 = sqrt(s11)

print('std =',std11)
Kar1 = 1-Char_val1*std11

print(Kar1)
plt.plot(L1, K1,'r.', label = 'Measurements spruce')
plt.plot(-0,Kar1,'mx',label = 'Characteristic value spruce')
#plt.xlabel('Logarithmic number of cycles (Log N)')
#plt.ylim((0, 2))
plt.xlim(-1,1.4)
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Static tests with characteristic value')
plt.xticks([])
plt.legend()
plt.show()
