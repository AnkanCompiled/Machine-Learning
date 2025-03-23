import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# Load dataset
california = fetch_california_housing()

# california.data contains the feature values and california.feature_names provides column names
#type of california.data, california.feature_names and california.target are numpy.ndarray
df = pd.DataFrame(california.data, columns=california.feature_names)
 # Target (median house value in $100,000s e.g., 2.5 means $250,000)
df['MedHouseVal'] = california.target 

# Features 
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# Target variable 
y = df['MedHouseVal']  

# test_size=0.2: 20% of data is used for testing.
# random_state=42: Ensures the split is the same every time the code runs (reproducibility).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
# Trains the model using the training data
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model Lower MSE = Better model performance.
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


