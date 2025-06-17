# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start: Load the dataset with house features, prices, and occupant counts.
2. Preprocess Data: Handle missing values, normalize features, and split data into training and test sets.
3. Initialize Model: Create an SGDRegressor with suitable hyperparameters (e.g., learning rate, epochs).
4.Train Model: Fit the regressor to the training data using stochastic gradient descent.
5.Predict & Evaluate: Use the trained model to predict house prices and occupant counts, then evaluate accuracy.
6.End.

## Program:
```python
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Guru Raghav Ponjeevith
RegisterNumber:  212223220027
*/
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data = fetch_california_housing()
x = data.data[:, :3]
y = np.column_stack((data.target, data.data[:, 6]))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train, y_train)
y_pred = multi_output_sgd.predict(x_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)
print("Predictions:\n", y_pred)
print("True values:\n", y_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

## Output:
![{CE6FD013-6CB0-4F70-8CE9-970D81124928}](https://github.com/user-attachments/assets/a8c0d279-f12f-4773-b800-b62d2cfa5e94)




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
