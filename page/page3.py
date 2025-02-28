import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
def load_data():
    car_data = pd.read_csv(r'C:\Users\s6404062620087\Desktop\IS_project\data\Car_Data_with_Missing_Values.csv')
    ddos_data = pd.read_csv(r'C:\Users\s6404062620087\Desktop\IS_project\data\DDoS_Dataset_with_Missing_Values.csv')
    return car_data, ddos_data
car_data, ddos_data = load_data()
def app():
    st.write("### การจัดการค่าที่หายไปในข้อมูลรถยนต์")

    # โค้ดการเติมค่าที่หายไป
    st.code("""
    # เติมค่าที่หายไปใน Car Data (ใช้ SimpleImputer สำหรับค่าตัวเลข)
    imputer = SimpleImputer(strategy='mean')  # เติมค่าที่หายไปด้วยค่าเฉลี่ย
    car_data['Year'] = imputer.fit_transform(car_data[['Year']])  # แทนที่ค่าที่หายไปในคอลัมน์ Year
    car_data['Present_Price'] = imputer.fit_transform(car_data[['Present_Price']])  # แทนที่ค่าที่หายไปในคอลัมน์ Present_Price
    car_data['Kms_Driven'] = imputer.fit_transform(car_data[['Kms_Driven']])  # แทนที่ค่าที่หายไปในคอลัมน์ Kms_Driven
    """)

    # เติมค่าที่หายไปใน Car Data (ใช้ SimpleImputer สำหรับค่าตัวเลข)
    imputer = SimpleImputer(strategy='mean')  # เติมค่าที่หายไปด้วยค่าเฉลี่ย
    car_data['Year'] = imputer.fit_transform(car_data[['Year']])  # แทนที่ค่าที่หายไปในคอลัมน์ Year
    car_data['Present_Price'] = imputer.fit_transform(car_data[['Present_Price']])  # แทนที่ค่าที่หายไปในคอลัมน์ Present_Price
    car_data['Kms_Driven'] = imputer.fit_transform(car_data[['Kms_Driven']])  # แทนที่ค่าที่หายไปในคอลัมน์ Kms_Driven

    # แสดงค่าที่หายไปหลังการเติมค่าใน Car Data
    st.write("### ค่าที่หายไปหลังการเติมค่าใน Car Data:")
    st.write(car_data.isnull().sum())

    # การแปลงข้อมูลประเภทข้อความใน Car Data
    st.write("### การแปลงข้อมูลที่เป็นข้อความในข้อมูลรถยนต์")

    # โค้ดการแปลงข้อมูลประเภทข้อความเป็นตัวเลข
    st.code("""
    # การแปลงข้อมูลประเภทข้อความใน Car Data
    label_encoder = LabelEncoder()
    car_data['Fuel_Type'] = label_encoder.fit_transform(car_data['Fuel_Type'])  # แปลง Fuel_Type
    car_data['Seller_Type'] = label_encoder.fit_transform(car_data['Seller_Type'])  # แปลง Seller_Type
    car_data['Transmission'] = label_encoder.fit_transform(car_data['Transmission'])  # แปลง Transmission
    """)

    label_encoder = LabelEncoder()
    car_data['Fuel_Type'] = label_encoder.fit_transform(car_data['Fuel_Type'])  # แปลง Fuel_Type
    car_data['Seller_Type'] = label_encoder.fit_transform(car_data['Seller_Type'])  # แปลง Seller_Type
    car_data['Transmission'] = label_encoder.fit_transform(car_data['Transmission'])  # แปลง Transmission

    # แสดงตัวอย่างข้อมูลที่แปลงแล้วใน Car Data
    st.write("### ตัวอย่างข้อมูลที่แปลงแล้วใน Car Data:")
    st.write(car_data.head())

    # สรุปการจัดการข้อมูล
    st.write("""
    ### สรุปการจัดการข้อมูลในข้อมูลรถยนต์ (Car Data):

    1. **การจัดการค่าที่หายไป (Missing Values)**:
    - ใช้ **`SimpleImputer(strategy='mean')`** เพื่อเติมค่าที่หายไปในคอลัมน์ตัวเลข ได้แก่ **`Year`**, **`Present_Price`**, และ **`Kms_Driven`** โดยเติมค่าที่หายไปด้วย **ค่าเฉลี่ย** (Mean) ของคอลัมน์นั้น ๆ

    2. **การตรวจสอบค่าที่หายไปหลังการเติมค่า**:
    - คอลัมน์ที่มีค่าหายไปในข้อมูลถูกเติมค่าด้วยค่าเฉลี่ยจากข้อมูลที่มีอยู่
    - ผลลัพธ์จากการเติมค่าที่หายไปจะสามารถตรวจสอบได้จากการใช้ **`isnull().sum()`** ซึ่งแสดงผลว่าค่าที่หายไปในคอลัมน์เหล่านั้นได้รับการเติมค่าด้วยค่าเฉลี่ยแล้ว

    3. **การแปลงข้อมูลที่เป็นข้อความ (Categorical Data) ให้เป็นตัวเลข (Label Encoding)**:
    - ใช้ **`LabelEncoder`** ในการแปลงข้อมูลประเภท **ข้อความ** เช่น **`Fuel_Type`**, **`Seller_Type`**, และ **`Transmission`** ให้เป็นตัวเลข เพื่อให้โมเดล Machine Learning สามารถใช้งานได้
    - การแปลงนี้ช่วยให้ข้อมูลมีรูปแบบที่เหมาะสมในการนำไปใช้ในขั้นตอนต่อไปของการสร้างโมเดล

    4. **ตัวอย่างข้อมูลที่แปลงแล้ว**:
    - คอลัมน์ที่เป็น **ข้อความ** เช่น **`Fuel_Type`**, **`Seller_Type`**, และ **`Transmission`** ถูกแปลงเป็น **ตัวเลข** โดยใช้ **`LabelEncoder`** เพื่อให้สามารถใช้งานได้ในโมเดล Machine Learning
    """)
    st.write("### การจัดการค่าที่หายไปในข้อมูลการโจมตี DDos")
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

    # แสดงโค้ดที่ใช้ในการจัดการข้อมูล
    st.write("### โค้ดที่ใช้ในการจัดการข้อมูล:")
    st.code("""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder

    # มองข้ามคอลัมน์ที่ไม่ต้องการ
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
    st.write(columns_to_consider.head())
    """)

    # สรุปการจัดการข้อมูล
    st.write("""
    ### สรุปการจัดการข้อมูลในข้อมูล DDoS:

    1. **การจัดการค่าที่หายไป (Missing Values)**:
    - ใช้ **`SimpleImputer(strategy='mean')`** เพื่อเติมค่าที่หายไปในคอลัมน์ที่เป็นข้อมูลตัวเลข ได้แก่ **`Source Port`**, **`Dest Port`**, **`Packet Length`**, และ **`Packets/Time`** โดยเติมค่าที่หายไปด้วย **ค่าเฉลี่ย** (Mean) ของคอลัมน์นั้น ๆ

    2. **การแปลงข้อมูลที่เป็นข้อความ (Categorical Data)**:
    - ใช้ **`LabelEncoder`** ในการแปลงข้อมูลที่เป็นประเภท **ข้อความ** ในคอลัมน์ **`target`** ให้เป็น **ตัวเลข** เพื่อให้สามารถใช้งานได้ในโมเดล Machine Learning

    3. **ตัวอย่างข้อมูลที่แปลงแล้ว**:
    - คอลัมน์ **`target`** ถูกแปลงเป็น **ตัวเลข** โดยใช้ **`LabelEncoder`** เพื่อให้สามารถใช้งานได้ในโมเดล Machine Learning

""")

