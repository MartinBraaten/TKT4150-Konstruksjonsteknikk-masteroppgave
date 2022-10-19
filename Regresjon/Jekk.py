'''
Created on 12. jan. 2021

@author: Martinskole
'''

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,pi, asin, sin
from scipy.stats import linregress
#from scipy.optimize import curve_fit
#import functools

import pandas as pd
#from numpy import arcsin, arctan
#from cmath import cos

data = pd.read_csv(r'D:\.master\Master\Tester\Mine\Test_10_full.asc',sep="\t",skiprows = 38, decimal=',',names = ['Time','Force','Displacement','Dritt'], dtype={'Time': float, 'Force': float,'Displacement': float,'Dritt': float})
#data.columns = ["Time", "Force", "Displacement"] # "4", "5", "6", "7", "8", "9", "10"] 
#print(data
lendata = data.shape[0]
#test 5
#start = 14620
#slutt = 248400
#Test 6
#start = 12600
#slutt = 66600

start = 37400
slutt = 56510200 #lendata
hertz = 6
F_amp = 12.52143
antall = 200 

hz = antall*hertz**-1

t = data.iloc[start:slutt,0].values  # values converts it into a numpy array
d = data.iloc[start:slutt,2].values  # -1 means that calculate the dimension of rows, but have 1 column
f = data.iloc[start:slutt,1].values
#front = data.iloc[start:slutt,3].values
#back = data.iloc[start:slutt,4].values

t = np.array(t)
d = np.array(d)
f = np.array(f)

#front = np.array(front)
#back = np.array(back)
#new_data = np_f.replace(data, 'HD\,', 'HD')
#a = np.char.replace(t,',', '.').astype(float)
#t = t.astype(float)
#d = d.astype(float)
#f = f.astype(float)
#front = front.astype(np.float)
#back = back.astype(np.float)


plt.plot(t, f,color='blue',label = 'Time-Force')

plt.title('Time - Force')
plt.ylabel('[kN]')
plt.xlabel('[s]')
#plt.legend()
#plt.savefig("Test13_F-t.pdf") #################################
#plt.show()

def finn_stigningstall():
    stigningstall = [] #np.zeros(int(round(slutt*antall**-1)))
    tid = [] #np.zeros_like(stigningstall)
    liten_liste_x = []
    liten_liste_y = []
    tid_a = []
    
    
    for i in range(0, len(d)): #int(round(slutt*antall**-1))):
        if (len(liten_liste_x) < hz):
            liten_liste_x.append(d[i])
            liten_liste_y.append(f[i])
            tid_a.append(t[i])
        else:
            ling = linregress(liten_liste_x, liten_liste_y)
            stigningstall.append(ling[0])
            tid.append(np.average(tid_a))
            liten_liste_x = []
            liten_liste_y = []
            tid_a = []
    #print(stigningstall)
    #print(len(stigningstall))
    #print(tid)
    derivert = np.zeros_like(tid)
    for i in range(0, len(stigningstall)):
        derivert[i] = (stigningstall[i]-stigningstall[i-1])/(tid[i]-tid[i-1])
    #print(derivert
    print('Endring per sykel =', np.average(derivert)
    print('Stig_1 =', stigningstall[1]
    print('Stig_-20 =', stigningstall[len(stigningstall)-20]
    print('Endring =', stigningstall[1]-stigningstall[len(stigningstall)-20]
    
    
    
    plt.plot(tid, stigningstall,color='red',label = 'Stiffness')

    plt.title('Time - Stiffness')
    plt.ylabel('[kN/mm]')
    plt.xlabel('[s]')
    #plt.legend()
    #plt.savefig("a.pdf") #################################
    plt.show()

finn_stigningstall()

X2 = f#[7620:220000]
Y2 = d#[7620:220000] #220000
lin = linregress((X2, Y2))
A = lin[0]
B = lin[1]


#print('A=',A, 'b=',B, 'R2=',lin[2]**2
print('k=',1/lin[0], 'B=',lin[1]/-lin[0], 'R2=',lin[2]**2
#y_mean = sum(Y2)/float(len(Y2))
#SStot = sum( [(x - y_mean)**2 for x in Y2])
#SSres = sum( [(y - (A*x + B))**2 for x, y in zip(X2, Y2) ])
#R2 = 1 - SSres/SStot
#print('R2 =',R2

#s = ((len(Y2)-1))**(-1) * SSres
#std = sqrt(s)
#print('s^2 =',s
#print('s =', std
#print('2s =', std*2

#xax = np.linspace(-25,25)

#plt.plot(lin[0]*xax + lin[1],xax, color='red',label = 'Regressed curve')
#plt.plot(Y2,X2)
#plt.title('Force - Displacement')
#plt.ylabel('[kN]')
#plt.xlabel('[mm]')
#plt.ylim((5, 29)) 
#plt.xlim((0.1,0.6))
#plt.legend()
#plt.savefig("Test13_F-u.pdf") ###################################
#plt.show()





##### Energi og demping ############




E = 0
E_tot = 0
lenx = len(X2)
n = (lenx*hz**-1)

for i in range(lenx-1):
    dE = 0.5*(X2[i+1]+X2[i])*(Y2[i+1]-Y2[i])
    E += dE*n**-1
    E_tot += dE
    
print('E =',E,'J'
print('E_tot =', E_tot


ksi = E * A**-1 * (2*pi * (F_amp)**2)**-1
print('ksi =',ksi

#Front = front[7620:7820]
#Back = back[7620:7820]

#Front = Front + 54.25
#Back = Back + 54.25

#alpha = arctan(12*(49**-1))


#for i in range(len(Front)-1):
#    H = sqrt((Front[i]*0.5)**2 + (Back[i]*0.5)**2 - 0.5*Front[i]*Back[i]*cos(alpha))
#    beta = asin(Back[i]*0.5 * H**-1 * sin(alpha))
    




