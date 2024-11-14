import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('modeling data.csv')

# Define features and target variables
X = df[[
    "S1 poke event", "S2 poke event", "M1 poke event", "M2 poke event",
    "M3 poke event", "Sp1 corner poke event", "Sp2 corner poke event", "Door event",
    "Match Box event", "Inactive event", "S1 poke duration", "S2 poke duration",
    "M1 poke duration", "M2 poke duration", "M3 poke duration", "Sp1 corner poke duration",
    "Sp2 corner poke duration", "Door duration", "Match Box duration", "Inactive duration",
    "port_pokes", "port_time", "corner_pokes", "corner_time"
]]

# List of target variables
target_metrics = [
    'Latency to corr sample', 'Latency to corr match', 
    'Num pokes corr sample', 'Time in corr sample', 'Num pokes inc sample',
    'Time in inc sample', 'False pos inc sample', 'Num pokes corr match',
    'Time in corr match', 'Num pokes inc match 1', 'Time in inc match 1',
    'False pos inc match 1', 'Num pokes inc match 2', 'Time in inc match 2',
    'False pos inc match 2',
]

# Function to train and evaluate Random Forest model
def train_rf_model(X, y, target_name):
    # Create binary target variable based on threshold (85th percentile)
    threshold = np.percentile(y, 85)
    y_binary = (y > threshold).astype(int)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hyperparameter tuning for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_scaled, y_binary)
    best_rf = grid_search.best_estimator_

    # Train Gradient Boosting classifier
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_scaled, y_binary)

    # Save the models, scaler, and features
    joblib.dump(best_rf, f'rf_model_{target_name}.pkl')
    joblib.dump(gb, f'gb_model_{target_name}.pkl')
    joblib.dump(scaler, f'scaler_{target_name}.pkl')
    joblib.dump(X.columns.tolist(), f'features_{target_name}.pkl')

    return best_rf, gb, scaler, y_binary

# Train and save ensemble models for each target variable
for metric in target_metrics:
    y = df[metric]

    # Train and evaluate Random Forest and Gradient Boosting models
    best_rf, gb_model, scaler, y_binary = train_rf_model(X, y, metric)

    # Create and train ensemble model
    voting_clf = VotingClassifier(estimators=[
        ('rf', best_rf),
        ('gb', gb_model)
    ], voting='soft')

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    voting_clf.fit(X_train_scaled, y_train)
    y_pred = voting_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Ensemble model accuracy for {metric}: {accuracy:.4f}")

    # Save the ensemble model
    joblib.dump(voting_clf, f'ensemble_model_{metric}.pkl')

print("All models trained, evaluated, and saved successfully.")
