import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import os
import numpy as np

def app():
    # **DDoS Prediction App**

# File paths
# The application needs the following file paths for data and model loading:
# - file_path: the location of the DDoS dataset.
# - model_path: the pre-trained model file.
# - image_path: an optional image to display in the app.

    file_path = r'data/DDoS_Dataset_with_Missing_Values.csv'
    model_path = r'page/model.h5'
    image_path = r'page/download.png'

    # Load dataset
    if not os.path.exists(file_path):
        st.error(f"Dataset not found at: {file_path}")
        return

    ddos_data = pd.read_csv(file_path)

    # **Load the Dataset**
    # The dataset is read into a Pandas DataFrame for preprocessing and analysis.
    # If the dataset is not found, the application stops and displays an error message.

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

    # อธิบายขั้นตอนการ Train โมเดล
    st.write("""
    #### ขั้นตอนที่ 1: การจัดการข้อมูล
    เราเริ่มต้นด้วยการจัดการค่าหายไปในคอลัมน์ข้อมูลเชิงตัวเลข โดยใช้ค่าเฉลี่ยเติมเต็มในแต่ละคอลัมน์ รวมถึงการแปลงค่าประเภทข้อความใน target เป็นค่าตัวเลขเพื่อให้ง่ายต่อการใช้งานในโมเดล
    """)

    st.code("""
    # เติมค่าที่หายไปในข้อมูลเชิงตัวเลข
    imputer = SimpleImputer(strategy='mean')
    columns_to_consider[['Source Port', 'Dest Port', 'Packet Length', 'Packets/Time']] = imputer.fit_transform(
        columns_to_consider[['Source Port', 'Dest Port', 'Packet Length', 'Packets/Time']]
    )

    # แปลงข้อมูลประเภทข้อความใน target เป็นตัวเลข
    label_encoder = LabelEncoder()
    columns_to_consider['target'] = label_encoder.fit_transform(columns_to_consider['target'])
    """, language="python")

    st.write("""
    #### ขั้นตอนที่ 2: การแยกข้อมูล
    ในขั้นตอนนี้ เราจะแยกข้อมูลออกเป็นส่วนที่เป็น Features (X) และส่วนที่เป็น Target (y) จากนั้นจะแบ่งข้อมูลเป็นชุดฝึก (Training Set) และชุดทดสอบ (Test Set)
    """)

    st.code("""
    # แยกข้อมูลเป็น X (features) และ y (target)
    X = columns_to_consider.drop(columns=['target'])
    y = columns_to_consider['target']

    # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """, language="python")

    st.write("""
    #### ขั้นตอนที่ 3: การสร้างและฝึกโมเดล
    เราจะใช้โมเดล Neural Network ซึ่งประกอบด้วยเลเยอร์หลายเลเยอร์ โดยเลเยอร์แรกจะมีจำนวนนิวรอนเท่ากับจำนวนฟีเจอร์ในข้อมูล และเลเยอร์สุดท้ายจะมีเพียง 1 นิวรอนพร้อมฟังก์ชัน Sigmoid เพื่อการ Classification โมเดลจะถูกฝึกด้วยค่า Binary Crossentropy Loss และ Optimizer แบบ Adam
    """)

    st.code("""
    # สร้าง Neural Network Model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # ชั้นแรก 64 นิวรอน
    model.add(Dense(32, activation='relu'))  # ชั้นที่สอง 32 นิวรอน
    model.add(Dense(1, activation='sigmoid'))  # เลเยอร์สุดท้ายสำหรับ Classification (2 คลาส)

    # คอมไพล์โมเดล
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # ฝึกโมเดล
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    """, language="python")

    st.write("""
    ขั้นตอนเหล่านี้ช่วยให้โมเดลสามารถเรียนรู้จากชุดข้อมูลฝึก และตรวจสอบประสิทธิภาพของมันด้วยชุดข้อมูลทดสอบ ซึ่งจะช่วยให้เราสามารถทำนายข้อมูลใหม่ได้อย่างแม่นยำ
    """)
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
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return

    model = load_model(model_path)
    
    # **Display Analysis Image**
    # If an image file is available, it will be displayed in the app.
    # This is purely for visual context and does not affect the predictions.
    if os.path.exists(image_path):
        st.image(image_path, caption="DDoS Attack Analysis", use_container_width=True, output_format='auto')
    else:
        st.error(f"Image file not found at: {image_path}")
    # Predict
        # Display evaluation metric (accuracy)
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

# Display predictions



