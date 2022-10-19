'''
Created on 5. mar. 2021

@author: Martinskole
'''
import numpy as np
import matplotlib.pyplot as plt

a = 6.7
b = 1.3
beta = 1
t = 100
R = -1


N = np.linspace(0,1, 10000000)
N_1 = np.linspace(1,500000)
k_fat = 1 - (1-R)/(a*(b-R))*np.log10(beta*N[1:-1]*t)
k_fat_1 = 1 - (1-R)/(a*(b-R))*np.log10(beta*N_1*t)

plt.plot(np.log10(N[1:-1]), k_fat,color='red') #np.log10(
plt.plot(np.log10(N_1), k_fat_1,color='red')
plt.xlabel('Log N')
plt.ylabel('Normalized load')
#plt.ylim((0, 1)) 
plt.xlim((-0.6,6))
plt.legend()
plt.show()



