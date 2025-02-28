import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model  # ใช้ load_model เพื่อโหลดไฟล์ .h5
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def app():
    # โหลดข้อมูล
    file_path = (r'C:\Users\s6404062620087\Desktop\IS_project\data\DDoS_Dataset_with_Missing_Values.csv')
    ddos_data = pd.read_csv(file_path)

    # การจัดการค่าที่หายไปในข้อมูลการโจมตี DDoS
    columns_to_consider = ddos_data.drop(columns=['Highest Layer', 'Transport Layer', 'Source IP', 'Dest IP'])

    # เติมค่าที่หายไปในข้อมูลเชิงตัวเลข
    imputer = SimpleImputer(strategy='mean')  # เติมค่าที่หายไปด้วยค่าเฉลี่ย
    columns_to_consider[['Source Port', 'Dest Port', 'Packet Length', 'Packets/Time']] = imputer.fit_transform(
        columns_to_consider[['Source Port', 'Dest Port', 'Packet Length', 'Packets/Time']]
    )

    # แปลงข้อมูลประเภทข้อความใน target เป็นตัวเลข
    label_encoder = LabelEncoder()
    columns_to_consider['target'] = label_encoder.fit_transform(columns_to_consider['target'])

    # แสดงผลข้อมูลที่เติมค่าหายไปแล้ว
    st.write("### ข้อมูลที่เติมค่าหายไปและแปลงข้อมูลแล้ว:")
    st.write(columns_to_consider.head())
    
    # แยกข้อมูลเป็น X (features) และ y (target)
    X = columns_to_consider.drop(columns=['target'])
    y = columns_to_consider['target']

    # แบ่งข้อมูลเป็นชุดฝึก (Training Set) และชุดทดสอบ (Test Set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # โหลดโมเดลที่ฝึกแล้วจากไฟล์ .h5
    model = load_model(r'C:\Users\s6404062620087\Desktop\IS_project\page\model.h5')  # เปลี่ยนจาก NN.pb เป็น model.h5

    # ทำนายผลลัพธ์
    y_pred = model.predict(X_test)

    # ประเมินผลลัพธ์
    st.write("### Prediction Results:")
    st.write(f"First 5 Predictions: {y_pred[:5]}")
    image_path = (r'C:\Users\s6404062620087\Desktop\IS_project\page\download.png')
    st.image(image_path)
    # แสดงผลลัพธ์การทำนาย
    accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Test Accuracy: {accuracy[1]:.4f}")

