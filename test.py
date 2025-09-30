import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from xgboost import XGBRegressor

# Load the dataset
data = pd.read_csv("modified_mart_success (1).csv")
data.columns = data.columns.str.strip()  # Remove extra spaces in column names

# Features and target
features = ['Item', 'Inventory', 'MRP', 'Sales', 'Weekly Sale']
target = 'Success Rate (%)'

# Encoding categorical features
if data['Item'].dtype == 'object':
    data['Item'] = data['Item'].astype('category').cat.codes

# Load the trained model
model_filename = "dmart_success.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Split data
X = data[features]
y_actual = data[target]

# Predict using the trained model
y_pred = model.predict(X)

# 1️⃣ **Feature Distribution - Histograms & Boxplots**
plt.figure(figsize=(15, 8))

for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)  # Ensure grid layout fits all features
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f"Histogram of {col}")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))

for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)  # Same grid fix for boxplots
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

# 2️⃣ **Correlation Heatmap**
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 3️⃣ **Residual Plot - Checking for Overfitting**
residuals = y_actual - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Success Rate")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()
