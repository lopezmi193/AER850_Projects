"""
AER850 Project 1
author: Miguel Lopez
501035749
"""
#Background and introduction
"""
Augmented Reality (AR) based instruction modules through predictive machine learning
(ML) algorithms has been a growing field of study to make AR modules more efficient and accurate 
primarily for the manufacturing and maintenance sectors within the aerospace industry. 

This project allows for exploring various classification-based ML algorithms to successfully
predict the maintenance step/stage given a specific part and its coordinates. The part
utilized for this project is an inverter of the FlightMax Fill Motion Simulator.

Based on the test subject identified, 13 unique steps are defined within the process of dis-
assembling the inverter. Each step has precise X, Y and Z axis points. The features for
the ML model are the coordinates and the target variable is the step the coordinates belong
to. The project is split into seven stages to design and develop ML models to predict the
maintenance step based on the coordinates provided
"""
#%%
#Imported Packages      

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import joblib
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#%%
#Step 1: Read CSV and convert to Dataframe

csv_file = 'Project_1_Data.csv'
df = pd.read_csv(csv_file)
df = df.replace(0, -1) 
df = df.dropna()
df = df.reset_index(drop=True)
train_y = df['Step']
train_X = df.drop(columns = ["Step"])

#%%
#Step 2: Data Visualization

attributes = ["X", "Y", "Z","Step"]
pd.plotting.scatter_matrix(df[attributes]) 
#This scatter plot matrix visualizes pairwise relationships between X,Y,Z and Step
plt.savefig('Data Visualazations.png')
plt.show()

#%%
#Step 3: Correlation Analysis

correlation_matrix = df.corr()
print(correlation_matrix)
plt.title('Pearson Correlation Matrix')
sns.heatmap(np.abs(correlation_matrix), annot=True, cmap="mako")
plt.savefig('Correlation Matrix.png')
plt.show()
corr1 = np.corrcoef(df['X'], df['Step'])
print("X correlation with Step is: ", round(corr1[0,1],2))
#-0.76  indicates a strong negative correlation.As X increases, Step tends to decrease significantly.
corr1 = np.corrcoef(df['Y'], df['Step'])
print("Y correlation with Step is: ", round(corr1[0,1],2))
#0.29. This shows a weak positive correlation; as Y increases, Step tends to increase slightly.
corr1 = np.corrcoef(df['Z'], df['Step'])
print("Z correlation with Step is: ", corr1[0,1])
#Correlation is approximately 0.20. This indicates a weak positive correlation.

#%%
#Step 4.1: Classification Model Development/Engineering - Model Training

#Model 1 - Linear Regression
model1 = LinearRegression()
model1.fit(train_X, train_y)

linear_prediction = model1.predict(train_X)
accuracy_model1 = mean_absolute_error(linear_prediction, train_y)
print("Linear regression model Mean Absolute Error (MAE): ", round(accuracy_model1,2))

#%%

#Model 2 - Decision Tree Regressor
model2= DecisionTreeRegressor()
model2.fit(train_X, train_y)
decision_tree_prediction = model2.predict(train_X)
accuracy_model2 = mean_absolute_error(decision_tree_prediction, train_y)
print("Decision tree regression model Mean Absolute Error (MAE): ", round(accuracy_model2,2))

#%%
#Model 3 Random Forest Regressor
model3 = RandomForestRegressor(n_estimators=10, random_state=16) #Pre Grid Search
model3.fit(train_X, train_y)
random_forest_prediction = model3.predict(train_X)
accuracy_model3 = mean_absolute_error(random_forest_prediction, train_y)
print("Random forest regression model Mean Absolute Error (MAE): ", round(accuracy_model3,2))

#%%
#Step 4.2 - Utilizing grid search cross-validation.

#k-fold cross-Validation (Linear Regression)
scores_model1 = cross_val_score(model1, train_X, train_y, cv=5, scoring='neg_mean_absolute_error')
mae_model1 = -scores_model1.mean()
print("Grid Search Cross validation Mean Absolute Error (MAE) for the linear regression: ", round(mae_model1, 2))

#k-fold cross-Validation (Decision Tree)
scores_model2 = cross_val_score(model2, train_X, train_y, cv=5, scoring='neg_mean_absolute_error')
mae_model2 = -scores_model2.mean()
print("Grid Search Cross validation Mean Absolute Error (MAE) for the decision tree regression: ", round(mae_model2, 2))

#k-fold cross-Validation (Random Forest)
scores_model3 = cross_val_score(model3, train_X, train_y, cv=5, scoring='neg_mean_absolute_error')
mae_model3 = -scores_model3.mean()
print("The Grid Search Cross validation  Mean Absolute Error (MAE) for the random forest regression: ", round(mae_model3, 2))

#%%
#Step 4.3 - Finding the best Hyperparameters using gridsearchCV

#Model 1 - Linear Regression GridSearchCV 
param_grid = {'fit_intercept': [True, False]}
grid_search = GridSearchCV(model1, param_grid={}, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
print("Ideal Hyperparameters for Linear Regression: ", best_params)

#Model 2: Decision Tree GridSearchCV 
param_grid = {
    'criterion': ['friedman_mse', 'poisson', 'absolute_error', 'squared_error'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2', 'sqrt'],
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.01, 0.1]
}

grid_search = GridSearchCV(model2, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
print("Ideal Hyperparameters for Decision Tree Regression: ", best_params)
best_model2 = grid_search.best_estimator_

#Model 3: Random Forest GridSearchCV 
param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(model3, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
print("Ideal Hyperparameters for Random Forest Regression: ", best_params)
best_model3 = grid_search.best_estimator_

train_y = pd.DataFrame(train_y).iloc[:, 0]
linear_prediction = pd.DataFrame(linear_prediction).astype(int).iloc[:, 0]
decision_tree_prediction = pd.DataFrame(decision_tree_prediction).astype(int).iloc[:, 0]
random_forest_prediction = pd.DataFrame(random_forest_prediction).astype(int).iloc[:, 0]

#%%
#Step 5 - Model Performance Analysis

#Model 1 - Linear Regression
#Scores for Linear Regression
accuracy_linear = accuracy_score(train_y, linear_prediction)
precision_linear = precision_score(train_y, linear_prediction, average= 'weighted')
f1_linear = f1_score(train_y, linear_prediction, average= 'weighted')
print("Linear regression accuracy: ",round(accuracy_linear,2) )
print("Linear regression precision: ",round(precision_linear,2) )
print("Linear regression F1 score: ",round(f1_linear,2) )

#Confusion Matrix for Linear Regression
cm_linear = confusion_matrix(train_y, linear_prediction)
class_labels = [str(i) for i in range(1, 14)]
sns.heatmap(cm_linear, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="mako")
plt.title('Confusion Matrix - Linear Regression')
plt.xlabel('Predicted Step Value')
plt.ylabel('Actual Step Value')
plt.savefig('Confusion Matrix - Linear Regression.png')
plt.show()

#Model 2 - Decision Tree
#Scores for Decision Tree
accuracy_decision_tree = accuracy_score(train_y, decision_tree_prediction)
precision_decision_tree = precision_score(train_y, decision_tree_prediction, average= 'weighted')
f1_decision_tree = f1_score(train_y, decision_tree_prediction, average= 'weighted')
print("Decision tree regression accuracy: ",round(accuracy_decision_tree,2))
print("Decision tree regression precision: ",round(precision_decision_tree,2))
print("Decision tree regression F1 score: ",round(f1_decision_tree,2))

#Confusion Matrix for Decision Tree
cm_decision_tree = confusion_matrix(train_y, decision_tree_prediction)
class_labels = [str(i) for i in range(1, 14)]
sns.heatmap(cm_decision_tree, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="mako")
plt.title('Confusion Matrix - Decision Tree Regression')
plt.xlabel('Predicted Step Value')
plt.ylabel('Actual Step Value')
plt.savefig('Confision Matrix - Decision Tree Regression.png')
plt.show()

#Model 3 - Random Forest
#Scores for Random Forest
accuracy_random_forest = accuracy_score(train_y, random_forest_prediction)
precision_random_forest = precision_score(train_y, random_forest_prediction, average= 'weighted')
f1_random_forest = f1_score(train_y, random_forest_prediction, average= 'weighted')
print("The accuracy score for Model 3 is: ",accuracy_random_forest)
print("The precision score for Model 3 is: ",precision_random_forest)
print("The F1 score for Model 3 is: ",f1_random_forest)

#Confusion Matrix for Random Forest
cm_random_forest = confusion_matrix(train_y, random_forest_prediction)
class_labels = [str(i) for i in range(1, 14)]
sns.heatmap(cm_random_forest, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="mako")
plt.title('Confusion Matrix - Random Forest Regression')
plt.xlabel('Predicted Step Value')
plt.ylabel('Actual Step Value')
plt.savefig('Confusion Matrix - Random Forest Regression')
plt.show()

#%%
#Step 6: Stacked Model Performance Analysis

joblib.dump(model2, 'DecisionTreeRegression.joblib')
loaded_model2 = joblib.load('DecisionTreeRegression.joblib')
joblib.dump(model3, 'RandomForestRegression.joblib')
loaded_model3 = joblib.load('RandomForestRegression.joblib')


#Stacking Model 2 and Model 3
estimators = [('decision_tree', model2), ('random_forest', model3)]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(train_X, train_y)

# Predictions
stacking_predictions = stacking_model.predict(train_X)

# Analyzeperformance metrics
accuracy_stacking = accuracy_score(train_y, stacking_predictions)
precision_stacking = precision_score(train_y, stacking_predictions, average='weighted')
f1_stacking = f1_score(train_y, stacking_predictions, average='weighted')

# Print metrics
print("Stacked model accuracy: ", round(accuracy_stacking, 2))
print("Stacked model precision: ", round(precision_stacking, 2))
print("Stacked model F1 score: ", round(f1_stacking, 2))    

# Confusion matrix
cm_stacking = confusion_matrix(train_y, stacking_predictions)
class_labels = [str(i) for i in range(1, 14)]
sns.heatmap(cm_stacking, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="mako")
plt.title('Confusion Matrix - Stacked Model')
plt.xlabel('Predicted Step Value')
plt.ylabel('Actual Step Value')
plt.savefig('Confusion Matrix - Stacked Model.png')
plt.show()

 
joblib.dump(model2, 'DecisionTreeRegression.joblib')
loaded_model3 = joblib.load('DecisionTreeRegression.joblib')
validation_data = [[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]]
predicted_vals = loaded_model3.predict(validation_data)
print('The predicted values are: ', predicted_vals)
     