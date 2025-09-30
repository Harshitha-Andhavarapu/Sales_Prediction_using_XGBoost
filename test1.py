import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('final data set.csv')
print("Dataset Loaded Successfully")

# Strip any whitespace from column names
data.columns = data.columns.str.strip()

# Print available columns
print("Available columns:", data.columns.tolist())

# Selecting only the required features
features = ['Item', 'Inventory', 'MRP', 'Sales', 'Weekly Sale']
target = 'Success Rate (%)'  # Update target variable name

# Ensure required columns exist
if not all(col in data.columns for col in features + [target]):
    raise ValueError("Some required columns are missing in the dataset")

# Encoding categorical variables before handling missing values
if data['Item'].dtype == 'object':
    data['Item'] = data['Item'].astype('category').cat.codes

# Handle missing values
for col in features:
    if data[col].dtype in ['int64', 'float64']:
        data[col].fillna(data[col].median(), inplace=True)

# Splitting data into features and target
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
model_filename = 'dmart_success.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)
print(f"Model loaded successfully from {model_filename}")

# Predict on test (unseen) data
y_test_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"Mean Squared Error on Unseen Data: {mse}")
print(f"R-squared Score on Unseen Data: {r2}")

# Save predictions along with actual values
test_results = X_test.copy()
test_results['Actual Success Rate (%)'] = y_test
test_results['Predicted Success Rate (%)'] = y_test_pred

# Save to CSV for analysis
test_results.to_csv('test_predictions.csv', index=False)
print("Test predictions saved to 'test_predictions.csv'!")

# Display sample results
print(test_results.head())
