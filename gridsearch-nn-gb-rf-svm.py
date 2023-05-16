import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load the data into a Pandas DataFrame
data = pd.read_excel('fleurs-orange-data-2022.xlsx')

# Split the data into features and target variables
X = data.drop(['Y','Unnamed: 0'], axis=1)
y = data['Y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and tune the Random Forest model
rf_model = RandomForestRegressor()
rf_params = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = GridSearchCV(rf_model, rf_params, cv=5)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
rf_preds = rf_model.predict(X_test)

# Train and tune the Support Vector Regression model
svm_model = SVR()
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(svm_model, svm_params, cv=5)
svm_grid.fit(X_train, y_train)
svm_model = svm_grid.best_estimator_
svm_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
svm_preds = svm_model.predict(X_test)

# Train and tune the Gradient Boosting Regression model
gb_model = GradientBoostingRegressor()
gb_params = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
gb_grid = GridSearchCV(gb_model, gb_params, cv=5)
gb_grid.fit(X_train, y_train)
gb_model = gb_grid.best_estimator_
gb_scores = cross_val_score(gb_model, X_train, y_train, cv=5)
gb_preds = gb_model.predict(X_test)

# Train and tune the Neural Network Regression model
nn_model = MLPRegressor()
nn_params = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}
nn_grid = GridSearchCV(nn_model, nn_params, cv=5)
nn_grid.fit(X_train, y_train)
nn_model = nn_grid.best_estimator_
nn_scores = cross_val_score(nn_model, X_train, y_train)
