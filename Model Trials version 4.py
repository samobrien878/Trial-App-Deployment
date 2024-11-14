import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('modeling data.csv')

# Define target variables
target_variables = [
    'Latency to corr match'
]

def train_rf_model(X, y, target_name):
    # Create binary target variable
    threshold = np.percentile(y, 85)
    y_binary = (y > threshold).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.5, random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Train ensemble model
    top_models = [(grid_search.best_estimator_,)]
    ensemble_models = [(f"RF", model[0]) for model in top_models]

    # Evaluate ensemble model
    y_pred = grid_search.best_estimator_.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble model accuracy for {target_name}: {ensemble_accuracy:.4f}")

    return grid_search.best_estimator_

# Loop through target variables
for y_name in target_variables:
    y = df[y_name]
    X = df.drop(target_variables + [y_name], axis=1)

    # Train and evaluate RF model
    rf_model = train_rf_model(X, y, y_name)