# page1.py
import streamlit as st
import pandas as pd
def app():  # เพิ่มฟังก์ชัน app()
    st.write("### ภาพรวมของโปรเจค")
    st.write("โปรเจคนี้นำเสนอ Machine Learning และ Neural Network โดยใช้ Streamlit ในการสร้าง Web Application")
    
    # ตั้งค่าหัวข้อหลักของเว็บแอปพลิเคชัน
    st.title("การวิเคราะห์ข้อมูลรถยนต์จาก CarDekho Dataset")

    # แสดงที่มาของ Dataset
    st.header("ที่มาของ Dataset")
    st.markdown(
        "ข้อมูลที่ใช้ในโปรเจกต์นี้นำมาจาก Kaggle: "
        "[Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=car+data.csv)\n"
        "และมีการใช้ chat gpt ในการทำให้ข้อมุลบางส่วนหายไป"
    )
    st.write("### ข้อมูลตัวอย่างจาก Car Dataset:")
    car_data = pd.read_csv(r"C:\Users\s6404062620087\Desktop\IS_project\data\Car_Data_with_Missing_Values.csv")
    st.write(car_data.head())
    # อธิบาย Feature ของ Dataset
    st.header("Feature ของ Dataset")
    st.markdown(
        "Dataset นี้ประกอบไปด้วยข้อมูลรถยนต์ที่สามารถใช้วิเคราะห์ได้หลายมิติ โดยมี Feature หลักดังนี้:\n"
        "- **Car_Name**: ชื่อของรถยนต์\n"
        "- **Year**: ปีที่ผลิต\n"
        "- **Selling_Price**: ราคาขาย (แสดงเป็นหน่วยล้านบาท)\n"
        "- **Present_Price**: ราคาปัจจุบันของรถ (เป็นหน่วยล้านบาท)\n"
        "- **Kms_Driven**: ระยะทางที่รถถูกขับขี่ (กิโลเมตร)\n"
        "- **Fuel_Type**: ประเภทของเชื้อเพลิง (Benzine/Diesel/CNG)\n"
        "- **Seller_Type**: ประเภทของผู้ขาย (Dealer/Individual)\n"
        "- **Transmission**: ระบบเกียร์ (Manual/Automatic)\n"
        "- **Owner**: จำนวนเจ้าของรถก่อนหน้า\n"
    )

    # อธิบายความไม่สมบูรณ์ของ Dataset
    st.header("ความไม่สมบูรณ์ของ Dataset และแนวทางการเตรียมข้อมูล")
    st.markdown(
        "Dataset นี้มีความไม่สมบูรณ์ในบางส่วน เช่น:\n"
        "- ค่า Missing Values ในบางคอลัมน์ เช่น ราคาขาย หรือระยะทางที่ขับขี่\n"
        "- ค่า Outliers ที่อาจจะต้องมีการกำจัดหรือแก้ไข\n"
        "- ประเภทของข้อมูลที่อาจจะไม่เหมาะสม เช่น ค่าที่ควรเป็นตัวเลขแต่กลับอยู่ในรูปของข้อความ\n"
        "- ข้อมูลที่ซ้ำซ้อนกัน\n"
        "เพื่อให้สามารถนำข้อมูลมาใช้งานได้อย่างถูกต้อง จำเป็นต้องมีขั้นตอนการเตรียมข้อมูล (Data Preprocessing) เช่น การลบค่าที่ขาดหายไป, การกรองข้อมูลที่ผิดปกติ, และการปรับประเภทของข้อมูลให้ถูกต้อง"
    )
       # อธิบายความไม่สมบูรณ์ของ Dataset
    st.header("ที่มาของ DDoS Traffic Dataset")
    st.markdown(
        "ข้อมูลที่ใช้ในโปรเจกต์นี้เพิ่มเติมจาก Kaggle: "
        "[DDoS Traffic Dataset](https://www.kaggle.com/datasets/oktayrdeki/ddos-traffic-dataset)\n"
        "Dataset นี้ประกอบด้วยข้อมูลการจราจรเครือข่ายที่ใช้ในการวิเคราะห์การโจมตีแบบ DDoS (Distributed Denial of Service)"
    )
    st.write("### ข้อมูลตัวอย่างจาก DDoS Dataset")
    ddos_data = pd.read_csv(r"C:\Users\s6404062620087\Desktop\IS_project\data\DDoS_Dataset_with_Missing_Values.csv")
    st.write(ddos_data.head())
    # อธิบาย Feature ของ DDoS Traffic Dataset
    st.header("Feature ของ DDoS Traffic Dataset")
    st.markdown(
        "Dataset นี้ประกอบด้วยข้อมูลการจราจรบนเครือข่ายที่เกี่ยวข้องกับการโจมตี DDoS และการทำงานของเครือข่ายในสภาวะปกติ โดยฟีเจอร์หลักที่สำคัญประกอบไปด้วย:\n"
        "- **Protocol_Type**: ประเภทของโปรโตคอล (เช่น TCP, UDP, ICMP)\n"
        "- **Service**: ชนิดของบริการที่ใช้ (เช่น HTTP, FTP, SMTP)\n"
        "- **Flag**: สถานะของการเชื่อมต่อ (เช่น SF, REJ, S0)\n"
        "- **Src_bytes**: จำนวนไบต์ที่ส่งจากแหล่งข้อมูล\n"
        "- **Dst_bytes**: จำนวนไบต์ที่ส่งไปยังปลายทาง\n"
        "- **Land**: การโจมตีประเภท `land attack` (1 = ใช่, 0 = ไม่ใช่)\n"
        "- **Hot**: ค่าที่แสดงถึงการใช้งาน CPU สูง\n"
        "- **Count**: จำนวนการเชื่อมต่อที่เกิดขึ้นในระยะเวลาที่กำหนด\n"
        "- **Srv_count**: จำนวนการเชื่อมต่อที่ถูกทำในบริการเดียวกัน\n"
        "- **Rerror_rate**: อัตราการเกิดข้อผิดพลาดระหว่างการเชื่อมต่อ\n"
        "- **Serror_rate**: อัตราการเกิดข้อผิดพลาดจากปลายทาง"
    )

    st.write("\n")
    st.info("ในขั้นตอนถัดไป เราจะทำการสำรวจและเตรียมข้อมูลให้เหมาะสมก่อนเริ่มการวิเคราะห์และสร้างโมเดล")
