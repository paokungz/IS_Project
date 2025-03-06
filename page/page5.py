import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import numpy as np

# File paths
rffile = Path(__file__).parent
modelp = rffile/"model.h5"
img = rffile/"download.png"
dfile = Path(__file__).parent.parent/"data"
dpath = dfile/"DDoS_Dataset_with_Missing_Values.csv"

def app5():
    # **DDoS Prediction App**

    # Load dataset
    if not os.path.exists(dpath):
        st.error(f"Dataset not found at: {dpath}")
        return

    ddos_data = pd.read_csv(dpath)

    # **Drop Unnecessary Columns**
    # Certain columns in the dataset are not needed for prediction and are dropped.
    # This helps simplify the data and focus on relevant features.
    drop_cols = ['Highest Layer', 'Transport Layer', 'Source IP', 'Dest IP']
    ddos_data = ddos_data.drop(columns=drop_cols, errors='ignore')

    # **Handle Missing Values**
    # Missing values in numeric columns are replaced with the mean of the column.
    # This ensures that the model receives complete data for prediction.
    numeric_cols = ddos_data.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    ddos_data[numeric_cols] = imputer.fit_transform(ddos_data[numeric_cols])

    # **Encode Target Variable**
    # If the dataset includes a target variable (e.g., whether an attack occurred),
    # it is converted from categorical text to numeric labels.
    if 'target' in ddos_data.columns:
        label_encoder = LabelEncoder()
        ddos_data['target'] = label_encoder.fit_transform(ddos_data['target'])

    # **Show Processed Dataset**
    # Display the first few rows of the processed dataset so users can verify
    # that missing values were handled and the target variable was encoded.
    st.write("### ข้อมูลที่ผ่านการเติมค่าหายไปและแปลงข้อมูลเรียบร้อย")

    # Show dataset to user
    st.write(ddos_data.head())

    # **Split Dataset**
    # The data is split into training and testing sets.
    # This allows the model to learn on one portion and be evaluated on unseen data.
    if 'target' in ddos_data.columns:
        X = ddos_data.drop(columns=['target'])
        y = ddos_data['target']
    else:
        st.error("No 'target' column found in dataset!")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # **Load Pre-trained Model**
    # The trained model (saved as an H5 file) is loaded to make predictions.
    # If the model file is not found, the application stops and displays an error.
    if not os.path.exists(modelp):
        st.error(f"Model file not found at: {modelp}")
        return

    model = load_model(modelp)

    # **Display Analysis Image**
    # If an image file is available, it will be displayed in the app.
    # This is purely for visual context and does not affect the predictions.
    if os.path.exists(img):
        st.image(img, caption="DDoS Attack Analysis", use_container_width=True, output_format='auto')
    else:
        st.error(f"Image file not found at: {img}")

    # **Model Evaluation**
    accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write("# Test Accuracy: {:.4f}".format(accuracy[1]))

    # **Demo the Model with User Input**
    # Interactive input fields let users provide values for the model to predict.
    # The application displays a prediction (e.g., Yes/No) based on the user input.
    source_port = st.number_input("Source Port", value=80)
    dest_port = st.number_input("Destination Port", value=443)
    packet_length = st.number_input("Packet Length", value=1000)
    packets_per_time = st.number_input("Packets/Time", value=10)

    # Prepare input for model
    input_data = [[source_port, dest_port, packet_length, packets_per_time]]
    input_data = np.array(input_data)

    # Predict using model
    single_prediction = model.predict(input_data)

    # Display predictions
    st.write("### Model Prediction for Single Input:")
    st.write(f"## Prediction: {'Yes' if single_prediction[0] > 0.5 else 'No'}")
