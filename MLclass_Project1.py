from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# from sklearn.linear_model import SGDRegressor
# from scipy.stats import zscore
import pandas as pd
import matplotlib.pyplot as plt

# Loading the Dataset
Dataset = pd.read_excel('proj1Dataset.xlsx')
Dataset = Dataset.dropna()
# ------------------------------------------------------------------------------
X = Dataset['Weight'].values.reshape(-1, 1)
T = Dataset['Horsepower'].values.reshape(-1, 1)

plt.figure(1)
plt.plot(X, T, 'rX')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title('Matlab\'s "carbig" Dataset', fontweight="bold")

regressor = LinearRegression()
regressor.fit(X, T)  # training the algorithm

Y = regressor.predict(X)

plt.plot(X[:, 0], Y, 'b', linewidth=2.0)
plt.legend(['Dataset', 'LinearRegression'])

# ------------------------------------------------------------------------------
X = Dataset['Weight'].values.reshape(-1, 1)
T = Dataset['Horsepower'].values.reshape(-1, 1)

plt.figure(2)
plt.plot(X, T, 'rX')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title('Matlab\'s "carbig" Dataset', fontweight="bold")

regressor = Ridge()
regressor.fit(X, T)

Y = regressor.predict(X)

plt.plot(X[:, 0], Y, 'g', linewidth=2.0)
plt.legend(['Dataset', 'Ridge'])
# ------------------------------------------------------------------------------
# X = Dataset['Weight'].values.reshape(-1,1)
# T = Dataset['Horsepower'].values.reshape(-1,1)

# plt.figure(3)
# plt.plot(X,T,'rX')
# plt.xlabel('Weight')
# plt.ylabel('Horsepower')
# plt.title('Matlab\'s "carbig" Dataset',fontweight="bold")

# X_norm = zscore(X)

# regressor = SGDRegressor()
# regressor.fit(X,T)

# Y = regressor.predict(X)

# plt.plot(X[:,0],Y,'y',linewidth = 2.0)
# plt.legend(['Dataset','SGDRegressor'])
