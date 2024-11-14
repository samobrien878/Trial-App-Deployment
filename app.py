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
    'Latency to corr sample', 'Latency to corr match', 'Corr sample port num',
    'Num pokes corr sample', 'Time in corr sample', 'Num pokes inc sample',
    'Time in inc sample', 'False pos inc sample', 'Num pokes corr match',
    'Time in corr match', 'Num pokes inc match 1', 'Time in inc match 1',
    'False pos inc match 1', 'Num pokes inc match 2', 'Time in inc match 2',
    'False pos inc match 2',
]

# Streamlit UI setup
st.set_page_config(
                   page_icon="🐭",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)
st.markdown("\n\n\n\n\n# Rat Performance Prediction App")
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
    .stSelectbox select {
        background-color: #333333; /* Dark gray select box */
        color: #00ffff; /* Light blue text in select box */
        font-size: 16px;
        border: none;
        border-radius: 5px;
        padding: 10px;
    }
    .stTextInput input, .stNumberInput input {
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
    rf_model = joblib.load(model_filename)
    return rf_model

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
                    # Prepare input data for prediction
                    input_data = row[features].values.reshape(1, -1)

                    # Load the model and scaler
                    rf_model, scaler = load_model_and_scaler(target_variable)

                    # Scale the input data
                    input_scaled = scaler.transform(input_data)

                    # Make the prediction
                    prediction = rf_model.predict(input_scaled)

                    # Assign performance category
                    performance = "proficient" if prediction[0] == 0 else "lower performance"
                    rat_results[target_variable] = performance

                results.append(rat_results)

            # Display results as summary bubbles
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

# JavaScript for auto-updating sum fields
st.components.v1.html(
    """
    <script>
    function calculateSums() {
        // Get values from the input fields
        var s1PokeEvent = parseFloat(document.getElementById('S1 poke event').value) || 0;
        var s2PokeEvent = parseFloat(document.getElementById('S2 poke event').value) || 0;
        var m1PokeEvent = parseFloat(document.getElementById('M1 poke event').value) || 0;
        var m2PokeEvent = parseFloat(document.getElementById('M2 poke event').value) || 0;
        var m3PokeEvent = parseFloat(document.getElementById('M3 poke event').value) || 0;
        var sp1CornerPokeEvent = parseFloat(document.getElementById('Sp1 corner poke event').value) || 0;
        var sp2CornerPokeEvent = parseFloat(document.getElementById('Sp2 corner poke event').value) || 0;
        var s1PokeDuration = parseFloat(document.getElementById('S1 poke duration').value) || 0;
        var s2PokeDuration = parseFloat(document.getElementById('S2 poke duration').value) || 0;
        var m1PokeDuration = parseFloat(document.getElementById('M1 poke duration').value) || 0;
        var m2PokeDuration = parseFloat(document.getElementById('M2 poke duration').value) || 0;
        var m3PokeDuration = parseFloat(document.getElementById('M3 poke duration').value) || 0;
        var sp1CornerPokeDuration = parseFloat(document.getElementById('Sp1 corner poke duration').value) || 0;
        var sp2CornerPokeDuration = parseFloat(document.getElementById('Sp2 corner poke duration').value) || 0;

        // Calculate the sums
        var portPokes = s1PokeEvent + s2PokeEvent + m1PokeEvent + m2PokeEvent + m3PokeEvent;
        var cornerPokes = sp1CornerPokeEvent + sp2CornerPokeEvent;
        var pokeDuration = s1PokeDuration + s2PokeDuration + m1PokeDuration + m2PokeDuration + m3PokeDuration;
        var cornerPokeDuration = sp1CornerPokeDuration + sp2CornerPokeDuration;

        // Update the sum fields
        document.getElementById('port_pokes').value = portPokes.toFixed(2);
        document.getElementById('corner_pokes').value = cornerPokes.toFixed(2);
        document.getElementById('port_duration').value = pokeDuration.toFixed(2);
        document.getElementById('corner_duration').value = cornerPokeDuration.toFixed(2);
    }

    // Attach the calculateSums function to input events
    document.querySelectorAll('input').forEach(input => {
        input.addEventListener('input', calculateSums);
    });
    </script>
    """,
    height=0,
)

# Create input fields dynamically
input_data = {}
for feature in features:
    if feature not in ['port_pokes', 'corner_pokes', 'port_duration', 'corner_duration']:
        input_data[feature] = st.number_input(f'{feature}', key=feature)

# Compute sum fields
port_pokes = sum([input_data.get(f'{prefix} poke event', 0) for prefix in ['S1', 'S2', 'M1', 'M2', 'M3']])
corner_pokes = sum([input_data.get(f'Sp{num} corner poke event', 0) for num in ['1', '2']])
port_duration = sum([input_data.get(f'{prefix} poke duration', 0) for prefix in ['S1', 'S2', 'M1', 'M2', 'M3']])
corner_duration = sum([input_data.get(f'Sp{num} corner poke duration', 0) for num in ['1', '2']])

# Display sum fields
st.write(f'Port Pokes: {port_pokes}')
st.write(f'Corner Pokes: {corner_pokes}')
st.write(f'Port Duration: {port_duration}')
st.write(f'Corner Duration: {corner_duration}')

if st.button('Predict Individual Performance'):
    st.header('Individual Prediction Results')

    individual_results = {}
    for target_variable in target_metrics:
        # Prepare input data for prediction
        input_values = [input_data[feature] for feature in features]
        input_data_array = np.array(input_values).reshape(1, -1)

        # Load the model and scaler
        rf_model, scaler = load_model_and_scaler(target_variable)

        # Scale the input data
        input_scaled = scaler.transform(input_data_array)

        # Make the prediction
        prediction = rf_model.predict(input_scaled)

        # Assign performance category
        performance = "proficient" if prediction[0] == 0 else "lower performance"
        individual_results[target_variable] = performance

    # Display results for the individual rat
    for metric, performance in individual_results.items():
        st.write(f"{metric}: {performance}")
