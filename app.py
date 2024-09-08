import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

# Load the data
car_data = pd.read_csv('CarPrice_Assignment.csv')

# Exploratory Data Analysis (EDA)
# Check correlations
plt.figure(figsize=(10, 8))
sns.heatmap(car_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation between Features")
plt.show()

# Visualize the distribution of the target variable (price)
sns.histplot(car_data['price'], kde=True)
plt.title("Distribution of Car Prices")
plt.show()

# Identify skewness in numerical features
numeric_features = car_data.select_dtypes(include=[np.number]).columns
skewed_features = car_data[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)

# Plot skewness of top 5 skewed features
skewness = pd.DataFrame({'Skewness': skewed_features})
skewness = skewness[abs(skewness['Skewness']) > 0.5]
print(skewness)

# Handle missing values
# Check for missing values
print(car_data.isnull().sum())

# Since no missing values are shown, this is just a placeholder
# For missing numerical features, we can use:
car_data.fillna(car_data.mean(), inplace=True)

# Handle Categorical features using one-hot encoding
car_data_clean = car_data.drop(['car_ID', 'CarName'], axis=1)
categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'fuelsystem']
car_data_encoded = pd.get_dummies(car_data_clean, columns=categorical_cols, drop_first=True)

# Log transformation to reduce skewness for highly skewed features 
car_data_encoded['price'] = np.log1p(car_data_encoded['price'])

# Split data into features and target
X = car_data_encoded.drop('price', axis=1)
y = car_data_encoded['price']

# Handling Imbalanced Data with SMOTE
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")
