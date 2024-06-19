from sklearn.model_selection import train_test_split, KFold # type: ignore
import pandas as pd # type: ignore

# Load the dataset
data = pd.read_csv('CCPP_data.csv')

# Split data into features and target
x = data[['AT', 'V', 'AP', 'RH']]
y = data['PE']

# Train-Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Training Data")
print(x_train)
print(y_train)


print("Test Data")
print(x_test)
print(y_test)

#Linear Regression:

from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import cross_val_score # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore

# Initialize the model
lr = LinearRegression()

# Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lr_scores = cross_val_score(lr, x_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

print(f"Linear Regression CV RMSE: {-lr_scores.mean()}")

#Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor # type: ignore

# Initialize the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Cross-Validation
rf_scores = cross_val_score(rf, x_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

print(f"Random Forest Regressor CV RMSE: {-rf_scores.mean()}")

# Train the final model on the entire training set
rf.fit(x_train, y_train)

# Predict on the test set
y_pred = rf.predict(x_test)

# Evaluate the performance
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Test RMSE: {test_rmse}")

