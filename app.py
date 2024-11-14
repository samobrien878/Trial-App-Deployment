import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Define features and target variables
features = [
    "S1 poke event", "S2 poke event", "M1 poke event", "M2 poke event",
    "M3 poke event", "Sp1 corner poke event", "Sp2 corner poke event", "Door event",
    "Match Box event", "Inactive event", "S1 poke duration", "S2 poke duration",
    "M1 poke duration", "M2 poke duration", "M3 poke duration", "Sp1 corner poke duration",
    "Sp2 corner poke duration", "Door duration", "Match Box duration", "Inactive duration",
    "port_pokes","port_duration","corner_pokes","corner_duration"
]

# List of target variables
target_metrics = [
    'Time in corr sample', 'Time in inc sample', 'False pos inc sample',
    'Time in corr match', 'Time in inc match 1', 'False pos inc match 1',
    'Time in inc match 2', 'False pos inc match 2',
]

# Streamlit UI setup
st.set_page_config(page_title="Rat Behavior Assessment App",
                   page_icon="🐭",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)
st.markdown("\n\n\n\n\n")

# Custom CSS for black background and blue-themed elements
st.markdown(
    """
    <style>
    body, .main, .stApp {
        background-color: #000000; /* Black background */
        color: #00ffff; /* Light blue text */
    }
    h1, h2, h3, h4, h5, h6, .stText {
        color: #00ffff; /* Light blue headers and text */
    }
    .stButton>button {
        color: #ffffff;
        background-color: #0000ff; /* Blue buttons */
        border: none;
        padding: 12px 30px;
        font-size: 18px;
        cursor: pointer;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0073e6;
        transform: scale(1.1);
    }
    .stSelectbox select, .stTextInput input, .stNumberInput input {
        background-color: #333333; /* Dark gray input fields */
        color: #00ffff; /* Light blue text in input fields */
        font-size: 16px;
        border: none;
        border-radius: 5px;
        padding: 10px;
    }
    .stNumberInput label, .stTextInput label, .stSelectbox label {
        color: #00ffff; /* Light blue labels */
    }
    .block-container {
        padding-top: 0px; /* Remove padding to make top part black */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('Williams Lab. Habituation Behavior Assessment')

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Function to load the correct Random Forest model and scaler
def load_model(target_variable):
    model_filename = f'{target_variable.replace(" ", "_")}_rf_ensemble_model.pkl'
    try:
        rf_model = joblib.load(model_filename)
        return rf_model
    except FileNotFoundError:
        st.error(f"Model file for {target_variable} not found.")
        return None

# If a file is uploaded, process the data
if uploaded_file:
    uploaded_data = pd.read_csv(uploaded_file)

    if set(features).issubset(uploaded_data.columns):
        st.write("CSV file successfully loaded!")

        # Predict button
        if st.button('Predict'):
            st.header('Prediction Results')

            results = []

            # Iterate over each rat's data in the uploaded CSV
            for index, row in uploaded_data.iterrows():
                rat_results = {"Rat": index + 1}
                for target_variable in target_metrics:
                    rf_model = load_model(target_variable)
                    if rf_model:
                        input_data = row[features].values.reshape(1, -1)
                        prediction = rf_model.predict(input_data)
                        performance = "proficient" if prediction[0] == 0 else "lower performance"
                        rat_results[target_variable] = performance
                results.append(rat_results)

            # Display results
            for result in results:
                st.write(f"**Rat {result['Rat']}**")
                for metric, performance in result.items():
                    if metric != "Rat":
                        st.write(f"{metric}: {performance}")
                st.write("---")
    else:
        st.error("Uploaded CSV file does not contain the required columns.")

# Additional input fields for individual predictions
st.header('Input Habituation Data')

# Create input fields dynamically
input_data = {}
for feature in features:
    if feature not in ['port_pokes', 'corner_pokes', 'port_duration', 'corner_duration']:
        input_data[feature] = st.number_input(f'{feature}', key=feature)

# Compute derived fields
port_pokes = sum([input_data.get(f'{prefix} poke event', 0) for prefix in ['S1', 'S2', 'M1', 'M2', 'M3']])
corner_pokes = sum([input_data.get(f'Sp{num} corner poke event', 0) for num in ['1', '2']])
port_duration = sum([input_data.get(f'{prefix} poke duration', 0) for prefix in ['S1', 'S2', 'M1', 'M2', 'M3']])
corner_duration = sum([input_data.get(f'Sp{num} corner poke duration', 0) for num in ['1', '2']])

# Display derived fields
st.write(f'Port Pokes: {port_pokes}')
st.write(f'Corner Pokes: {corner_pokes}')
st.write(f'Port Duration: {port_duration}')
st.write(f'Corner Duration: {corner_duration}')

if st.button('Predict Individual Performance'):
    st.header('Individual Prediction Results')

    individual_results = {}
    for target_variable in target_metrics:
        input_values = [input_data.get(feature, 0) for feature in features]
        input_data_array = np.array(input_values).reshape(1, -1)
        rf_model = load_model(target_variable)
        if rf_model:
            prediction = rf_model.predict(input_data_array)
            performance = "proficient" if prediction[0] == 0 else "lower performance"
            individual_results[target_variable] = performance

    # Display results for the individual rat
    for metric, performance in individual_results.items():
        st.write(f"{metric}: {performance}")
