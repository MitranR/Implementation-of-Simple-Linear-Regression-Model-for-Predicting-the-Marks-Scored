# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students.

2.Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored.

3.Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crosses the y-axis.

4.Use the linear equation to predict marks based on the input Predicted Marks = m*(hours studied) + b.

5.For each data point calculate the difference between the actual and predicted marks.

6.Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error.

7.Once the model parameters are optimized, use the final equation to predict marks for any new input data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MITRAN R
RegisterNumber:  212224040192
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head(10)
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
##plotting for training data
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
##plotting for test data
plt.scatter(x_test,y_test,color="grey")
plt.plot(x_test,y_pred,color="purple")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:

Head:
<img width="1033" height="358" alt="image" src="https://github.com/user-attachments/assets/fc5b71a6-4b82-48da-9be3-bbe25c905546" />

Tail:
<img width="1031" height="207" alt="image" src="https://github.com/user-attachments/assets/f03362d6-3a37-4821-a3ae-3296b0486c85" />

X value:
<img width="1031" height="540" alt="image" src="https://github.com/user-attachments/assets/932fb782-3d1d-4166-a31e-2fb1ae4b94a8" />

Y Values:
<img width="1032" height="61" alt="image" src="https://github.com/user-attachments/assets/41c0a016-124e-4ef3-a7cd-172a7220633d" />

Y_Prediction Values:
<img width="1037" height="50" alt="image" src="https://github.com/user-attachments/assets/0b1158f9-beb6-4d76-bb0d-64024fae85ae" />

Y_Test Values:
<img width="1028" height="82" alt="image" src="https://github.com/user-attachments/assets/8f19c70f-e90e-48e5-b441-9f66d91535e9" />

MSE,MAE AND RMSE:
<img width="1036" height="77" alt="image" src="https://github.com/user-attachments/assets/3f3901d8-d1e5-4495-ac4a-0681e5f047c0" />

Training Set:
<img width="1035" height="630" alt="image" src="https://github.com/user-attachments/assets/629aa0a5-b1ff-4384-8132-26b440b22c8a" />

Testing Set:
<img width="1027" height="591" alt="image" src="https://github.com/user-attachments/assets/48ecb8f0-9dee-4400-8ba4-8a1d44a9798f" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
