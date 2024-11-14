import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import accuracy_score

df = pd.read_csv('modeling data.csv')
df = df[df['phase'] == 1]


# Select features and target variable
X = df[[
    "S1 poke event", "S2 poke event", "M1 poke event", "M2 poke event",
    "M3 poke event", "Sp1 corner poke event", "Sp2 corner poke event", "Door event",
    "Match Box event", "Inactive event", "S1 poke duration", "S2 poke duration",
    "M1 poke duration", "M2 poke duration", "M3 poke duration", "Sp1 corner poke duration",
    "Sp2 corner poke duration", "Door duration", "Match Box duration", "Inactive duration",
    "port_pokes", "port_time", "corner_pokes", "corner_time"
]]

y = df['Latency to corr sample']

threshold = np.percentile(y, 85)
y_binary = (y > threshold).astype(int)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit a Random Forest model to the entire dataset
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_scaled, y_binary)

# Extract feature importances
importances = forest.feature_importances_

# Create a DataFrame to visualize feature importances
feature_names = X.columns
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 8))
plt.barh(feature_importances['feature'], feature_importances['importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis()
plt.show()

# Select the best features based on feature importance
# You can set a threshold for the minimum importance you want to keep (e.g., top 20%)
top_features = feature_importances[feature_importances['importance'] > 0.05]
selected_features = top_features['feature'].tolist()

# Create a new DataFrame with the selected features
X_selected = X[selected_features]

# Optionally, re-scale the selected features if needed
X_selected_scaled = scaler.fit_transform(X_selected)

# Train the model with selected features
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_selected_scaled, y_binary)

# Save the model and the selected features
import joblib
joblib.dump(final_model, 'final_model.pkl')  # Save the trained model
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
joblib.dump(selected_features, 'selected_features.pkl')  # Save the list of selected features

print(f"Selected Features: {selected_features}")

selected_features = joblib.load('selected_features.pkl')
scaler = joblib.load('scaler.pkl')

# Assume df is the original dataset
X = df[selected_features]  # Use only selected features
y = df['Latency to corr sample']  # Ensure this is the correct target variable name

# Handle missing values if there are any
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

# Create the binary target variable (e.g., selecting the top 60% values)
threshold = np.percentile(y, 85)
y_binary = (y > threshold).astype(int)

# Standardize the features (using the scaler saved earlier)
X_scaled = scaler.transform(X)

# List to store accuracy for each trial
accuracies = []

# Perform 100 trials and track performance
n_trials = 100
for _ in range(n_trials):
    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.5, random_state=None)

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=4)  # You can adjust the n_neighbors parameter if needed
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Calculate the average accuracy over all trials
average_accuracy = np.mean(accuracies)
print(f"Average Accuracy over 100 trials: {average_accuracy * 100:.2f}%")

# Plot the accuracies of each trial
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_trials + 1), accuracies, marker='o', linestyle='-', color='b')
plt.title("KNN Model Accuracy over 100 Trials")
plt.xlabel("Trial Number")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


