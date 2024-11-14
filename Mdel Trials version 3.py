import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('modeling data.csv')

# Define target variables
target_variables = [
    'Latency to corr sample', 'Latency to corr match',
    'Time in corr sample',
    'Time in inc sample', 'False pos inc sample',
    'Time in corr match',  'Time in inc match 1',
    'False pos inc match 1', 'Time in inc match 2',
    'False pos inc match 2'
]

# Define feature subsets
feature_subsets = [
    ["S1 poke event", "S2 poke event", "M1 poke event", "M2 poke event", "M3 poke event"],
    ["Sp1 corner poke event", "Sp2 corner poke event", "Door event", "Match Box event", "Inactive event"],
    ["S1 poke duration", "S2 poke duration", "M1 poke duration", "M2 poke duration", "M3 poke duration"],
    ["Sp1 corner poke duration", "Sp2 corner poke duration", "Door duration", "Match Box duration", "Inactive duration"],
    ["port_pokes", "port_time", "corner_pokes", "corner_time"]
]

def train_rf_model(X, y, target_name):
    # Create binary target variable based on threshold (85th percentile)
    threshold = np.percentile(y, 85)
    y_binary = (y > threshold).astype(int)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.5, random_state=42)

    # Train models
    models = []
    for features in feature_subsets:
        features = [f for f in features if f in X_train.columns]
        if features:
            X_subset_train = X_train[features]
            X_subset_test = X_test[features]
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_subset_train, y_train)
            models.append((features, rf_model))

    # Evaluate model performance
    accuracies = []
    for features, model in models:
        y_pred = model.predict(X_test[features])
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Select top 5 models
    top_models = sorted(models, key=lambda x: accuracy_score(y_test, x[1].predict(X_test[x[0]])), reverse=True)[:5]

    # Create ensemble model
    ensemble_models = [(f"Model {i}", model[1]) for i, model in enumerate(top_models)]
    voting_clf = VotingClassifier(estimators=ensemble_models, voting='soft')

    # Train ensemble model
    features = set()
    for m in top_models:
        features.update(m[0])
    existing_features = list(features.intersection(X_train.columns))
    voting_clf.fit(X_train[existing_features], y_train)

    # Evaluate ensemble model
    y_pred = voting_clf.predict(X_test[existing_features])
    ensemble_accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble model accuracy for {target_name}: {ensemble_accuracy:.4f}")

    # Save the trained ensemble model
    model_filename = f"{target_name.replace(' ', '_')}_rf_ensemble_model.pkl"
    joblib.dump(voting_clf, model_filename)
    print(f"Model saved as {model_filename}")

    return voting_clf

# Loop through each target variable
for y_name in target_variables:
    y = df[y_name]
    X = df.drop(target_variables, axis=1)

    # Train and evaluate RF model
    rf_model = train_rf_model(X, y, y_name)
