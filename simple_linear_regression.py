import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression

#uploading the dataset
df_sal = pd.read_csv(r'C:\Users\TUF GAMING\Downloads\Salary_Data.csv')
df_sal.head()
df_sal.describe()

#data distribution
plt.title('Salary Distribution Plot')
sns.distplot(df_sal['Salary'])
plt.show()

plt.scatter(df_sal['YearsExperience'], df_sal['Salary'], color = 'lightcoral')
plt.title('Salary vs Experience')
plt.xlabel('Years Of Experiece')
plt.ylabel('Salary')
plt.box(False)
plt.show()

#splitting the variables
x = df_sal.iloc[:, :1] #independent
y = df_sal.iloc[:, 1:] #dependent

#splitting dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#regressor model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Prediction result
y_pred_test = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)

#prediction on the training dataset
plt.scatter (x_train, y_train, color = 'lightcoral')
plt.plot(x_train, y_pred_train, color = 'firebrick')
plt.title('Salary vs Experience(Training Set)')
plt.ylabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['x_train/y_train', 'x_train/Pred(y_test)'], title = "Sal/Exp", loc = 'best')
plt.box(False)
plt.show()

#prediction on the test set
plt.scatter(x_test, y_test, color = 'lightcoral')
plt.plot(x_train, y_pred_train, color = 'firebrick')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['X_train/y_train', 'x_train/Pred(y_test)'], title = 'Sal/Exp', loc = 'best')
plt.box(False)
plt.show()

#Regressor coefficients and intercept
print(f'Coefficient: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')