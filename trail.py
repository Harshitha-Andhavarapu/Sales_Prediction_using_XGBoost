import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('final data set.csv')
print("Dataset Loaded Successfully")

# Strip any whitespace from column names
data.columns = data.columns.str.strip()

# Print available columns
print("Available columns:", data.columns.tolist())

# Selecting only the required features (Dropped 'Item')
features = ['Inventory', 'MRP', 'Sales', 'Weekly Sale']
target = 'Success Rate (%)'

# Ensure required columns exist
if not all(col in data.columns for col in features + [target]):
    raise ValueError("Some required columns are missing in the dataset")

# Handle missing values
for col in features:
    if data[col].dtype in ['int64', 'float64']:
        data[col].fillna(data[col].median(), inplace=True)

# Splitting data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost with optimized hyperparameters
model = XGBRegressor(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=8, 
    colsample_bytree=1.0, 
    subsample=0.7, 
    objective="reg:squarederror",  # Ensures correct loss function
    verbosity=1, 
    random_state=42
)

# Cross-validation to estimate performance
cv_mse = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error").mean()
print(f"Cross-validation MSE: {cv_mse}")

# Train the model with early stopping (fixing the parameter issue)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * ((len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))

print(f"Optimized Mean Squared Error (MSE): {mse}")
print(f"Optimized Root Mean Squared Error (RMSE): {rmse}")
print(f"Optimized Mean Absolute Error (MAE): {mae}")
print(f"Optimized R-squared Score (R²): {r2}")
print(f"Optimized Adjusted R² Score: {adjusted_r2}")

# Convert Regression to Classification (For confusion matrix & precision)
thresholds = [33, 66]  # Define categories: Low (<33), Medium (33-66), High (>66)

def categorize(value):
    if value < thresholds[0]:
        return 0  # Low Success
    elif value < thresholds[1]:
        return 1  # Medium Success
    else:
        return 2  # High Success

# Apply categorization
y_test_class = np.array([categorize(val) for val in y_test])
y_pred_class = np.array([categorize(val) for val in y_pred])

# Classification Metrics
conf_matrix = confusion_matrix(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class, average='weighted')
accuracy = accuracy_score(y_test_class, y_pred_class)

print("\nClassification Metrics (Converted to Categories):")
print(f"Accuracy Score: {accuracy}")
print(f"Precision Score: {precision}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save the model
model_filename = 'dmart_success.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved successfully at {model_filename}")   
