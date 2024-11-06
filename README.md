Salary Prediction Using Linear Regression
This script demonstrates how to predict salary based on years of experience using linear regression. It involves data loading, visualization, model training, and evaluation. The primary goal is to build a regression model and use it to make predictions on both training and testing datasets.

Prerequisites
Before running this script, you need to install the following Python libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn

The dataset used in this script is Salary_Data.csv, which contains information about years of experience and corresponding salaries.

File Structure:
Salary_Data.csv: A CSV file containing two columns:
YearsExperience: The number of years of experience.
Salary: The corresponding salary.
Script Breakdown
Importing Libraries: The script imports necessary libraries for data manipulation, visualization, and machine learning:

pandas for data handling.
numpy for numerical operations.
matplotlib.pyplot and seaborn for data visualization.
sklearn.model_selection.train_test_split for splitting the data into training and testing sets.
sklearn.linear_model.LinearRegression for fitting the linear regression model.
Loading the Dataset: The dataset is loaded from the specified file path using pd.read_csv() and a quick look at the first few records is provided with df_sal.head(). Statistical summary of the dataset is shown using df_sal.describe().

Data Visualization:

Salary Distribution Plot: A histogram showing the distribution of salaries.
Scatter Plot of Salary vs. Years of Experience: A scatter plot showing the relationship between years of experience and salary.
Both plots are visualized using matplotlib and seaborn.

Data Preparation:

The independent variable (YearsExperience) is extracted as x and the dependent variable (Salary) as y.
The data is split into training and testing sets using train_test_split(). 80% of the data is used for training, and 20% is used for testing.
Model Training:

A linear regression model is instantiated using LinearRegression().
The model is trained on the training data (x_train, y_train) using the fit() method.
Prediction and Evaluation:

The model is used to predict salary values for both the training and testing datasets (y_pred_train and y_pred_test).
Scatter plots are used to visualize the model's performance:
Training Set Prediction: A scatter plot showing actual salary vs. years of experience for the training data, with the regression line overlaid.
Test Set Prediction: A similar plot for the test data.
Model Coefficients and Intercept:

The script prints the model’s regression coefficient and intercept values, which represent the slope and y-intercept of the fitted line.
Example Output
Visualizations:

A Salary Distribution Plot showing how the salary values are distributed.
A Salary vs Experience Scatter Plot illustrating the correlation between years of experience and salary.
Two regression plots (for the training and test data), showing the predicted vs actual salaries with the fitted regression line.
Model Coefficients: The model's coefficients and intercept will be printed, which can be used to interpret the linear relationship between years of experience and salary.
