import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # To visualize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sympy import symbols, solve

start = 1
slutt = 33
data = pd.read_excel(r'D:\.master\Master\malinger.xls', sheet_name='Furu plot')  # load data set
Y2 = data.iloc[start: slutt, 2].values  # values converts it into a numpy array
X2 = data.iloc[start: slutt, 1].values  # -1 means that calculate the dimension of rows, but have 1 column
X2 = np.array(X2)
Y2 = np.array(Y2)
X2 = X2.astype(float)
Y2 = Y2.astype(float)
#Y2 = Y2.reshape(-1,1)
X2 = X2.reshape(-1,1)

# Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(X2, Y2)
print('Slope of the line is', regression_model.coef_)
print('Intercept value is', regression_model.intercept_)

# Predict
y_predicted = regression_model.predict(X2)
xax = np.linspace(-1,10,10000)
plt.plot(y_predicted, X2, color ='g', label = 'n=1')
plt.plot((-0.6,7.623968605466832),(0.8995982784138086,0),color ='g')
# model evaluation
mse = mean_squared_error(Y2, y_predicted)

rmse = np.sqrt(mean_squared_error(Y2, y_predicted))
r2 = r2_score(Y2, y_predicted)

# printing values

print('MSE of Linear model', mse)
print('R2 score of Linear model: ', r2)
############################################
print('')
poly_features = PolynomialFeatures(degree = 2, include_bias = False)
x_poly = poly_features.fit_transform(X2)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, Y2)
k = lin_reg.intercept_

print('Coefficients of x are', lin_reg.coef_)
print('Intercept is', k)
k = lin_reg.intercept_

x_new = np.linspace(-1, 10, 10000).reshape(10000, 1)
x_new_poly = poly_features.transform(x_new)
y_new = lin_reg.predict(x_new_poly)
plt.plot(y_new, x_new, "r-", linewidth=2, label="n=2")
y_deg2 = lin_reg.predict(x_poly)
# model evaluation
mse_deg2 = mean_squared_error(Y2, y_deg2)
r2_deg2 = r2_score(Y2, y_deg2)
# printing values
print('MSE of Polyregression model', mse_deg2)
print('R2 score of Linear model: ', r2_deg2)

p = symbols('p')
expr = lin_reg.coef_[0]*p+lin_reg.coef_[1]*p**2+k
sol = solve(expr)
print('sol =',sol)
##############################################
print('')
poly_features1 = PolynomialFeatures(degree = 3, include_bias = False)
x_poly1 = poly_features1.fit_transform(X2)
lin_reg1 = LinearRegression()
lin_reg1.fit(x_poly1, Y2)
print('Coefficients of x are', lin_reg1.coef_)
print('Intercept is', lin_reg1.intercept_)

x_new1 = np.linspace(-1, 10, 10000).reshape(10000, 1)
x_new_poly1 = poly_features1.transform(x_new1)
y_new1 = lin_reg1.predict(x_new_poly1)
y_deg1 = lin_reg1.predict(x_poly1)
# model evaluation
mse_deg1 = mean_squared_error(Y2, y_deg1)
r2_deg1 = r2_score(Y2, y_deg1)
# printing values
print('MSE of Polyregression model', mse_deg1)
print('R2 score of Linear model: ', r2_deg1)

p = symbols('p')
expr = lin_reg1.coef_[0]*p+lin_reg1.coef_[1]*p**2+lin_reg1.intercept_ + lin_reg1.coef_[2]*p**3
sol = solve(expr)
print('sol =',sol)
##
plt.plot(Y2, X2, "b.")
plt.plot(y_new1, x_new1, "b-", linewidth=2, label="n=3")

plt.plot(-0.6,1,'rx', label ='f_u pine')
plt.xlim(-1,11)
plt.ylim(0,1.1)
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.legend()

plt.title(r'Pine: Polynomial regression for the $n$th degree')
plt.show()

#Y2 = Y2.reshape(-1,1)
#lin_reg.fit(x_poly, Y2)
#prediction = lin_reg.predict(Y2)
residual = (Y2 - y_deg2)

plt.plot(Y2,residual,'b.', label = 'Residuals')
plt.plot([0,7],[0,0])
plt.xlim(1,7)
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel('Residuals')
plt.title('Pine: \n Residual plot')
plt.legend()
plt.show()
############################################
'''

#x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
#y   = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
degree = 3

df = pd.DataFrame(columns=['Y2', 'X2'])
df['x'] = X2
df['y'] = Y2

weights = np.polyfit(X2, Y2, degree)
model = np.poly1d(weights)
results = smf.ols(formula='y ~ model(x)', data=df).fit()
results.summary()

xax = np.linspace(-1,11)

plt.plot(Y2, X2, 'b.', label = 'Measurements')
plt.plot(7,0.16,'+',color='black', label = 'Did not fail')
#plt.plot(model, xax, color='red',label = 'Regression curve')
plt.plot(-0.6,1,'bx',label = r'$f_{u}$ pine')
plt.ylim((0, 1.1))  #0, 1.1 # 0, 1.3
plt.xlim((-1, 8))
plt.xlabel('Logarithmic number of cycles (Log N)')
plt.ylabel(r'Normalized stress ($f_{a}/f_{u}$)')
plt.title('Pine: \n S-N curve based on fatigue loading')

plt.plot(-0.6,0.759194286,'rx',label = r'$f_{u}$ spruce')
plt.legend()
plt.show()
'''