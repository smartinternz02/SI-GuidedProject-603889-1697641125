# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import CuckooSearch, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load your cancer dataset
# Replace 'your_dataset.csv' with the actual file name or provide the path
dataset = pd.read_csv('your_dataset.csv')

# Assume your dataset has features (X) and labels (y)
X = dataset.drop('label_column', axis=1)
y = dataset['label_column']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using Cuckoo Search (CS)
cs = CuckooSearch()
cs.fit(X_train, y_train)
selected_features_cs = cs.selected_features_

# Feature selection using SelectFromModel with RandomForest (SMO)
rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
rf_selector.fit(X_train, y_train)
selected_features_smo = X_train.columns[rf_selector.get_support()]

# Combine selected features from CS and SMO
selected_features_combined = list(set(selected_features_cs) | set(selected_features_smo))

# Apply feature selection to the dataset
X_train_selected = X_train[selected_features_combined]
X_test_selected = X_test[selected_features_combined]

# Standardize the data
scaler = StandardScaler()
X_train_selected = scaler.fit_transform(X_train_selected)
X_test_selected = scaler.transform(X_test_selected)

# Build a Deep Learning model using TensorFlow/Keras
model = Sequential()
model.add(Dense(128, input_dim=X_train_selected.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_selected, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = (model.predict(X_test_selected) > 0.5).astype(int)
accuracy_dl = accuracy_score(y_test, y_pred)
print("Deep Learning Accuracy:", accuracy_dl)
