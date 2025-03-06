import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
def app():
    lfile = Path(__file__).parent.parent/"data"
    Lpath = lfile/"laptop_prices_mod.csv"
    dfile = Path(__file__).parent.parent/"data"
    dpath = dfile/"DDoS_Dataset_with_Missing_Values.csv"
    st.write("### ภาพรวมของโปรเจค")
    st.write("โปรเจคนี้นำเสนอ Machine Learning และ Neural Network โดยใช้ Streamlit ในการสร้าง Web Application")
    
    # เพิ่มการอธิบายข้อมูลของ Laptop ก่อนข้อมูล Traffic
    st.header("ที่มาของข้อมูลราคา Laptop")
    st.markdown(
        "ข้อมูลที่ใช้ในโปรเจกต์นี้เพิ่มเติมจาก Kaggle: "
        "[Laptop Prices Dataset](https://www.kaggle.com/datasets/asinow/laptop-price-dataset)\n"
        "Dataset นี้ประกอบด้วยข้อมูลของ Laptop หลายรุ่น โดยมีรายละเอียดที่สำคัญเช่น รุ่นของ Laptop, ขนาดหน้าจอ, ความจุ RAM, ความจุ Storage, และราคาของแต่ละรุ่น"
    )
    
    st.write("### ข้อมูลตัวอย่างจาก Laptop Prices Dataset")
<<<<<<< HEAD
    laptop_data = pd.read_csv(r"paokungz/IS_Project/blob/main/data/laptop_prices_mod.csv")
=======
    laptop_data = pd.read_csv(Lpath)
>>>>>>> eed7309 (Save changes before rebase)
    st.write(laptop_data.head())
    
    # อธิบาย Feature ของ Laptop Dataset
    st.header("Feature ของ Laptop Prices Dataset")
    st.markdown(
        "Dataset นี้ประกอบด้วยข้อมูลรายละเอียดของ Laptop โดยฟีเจอร์หลักที่สำคัญประกอบไปด้วย:\n"
        "- **Brand**: แบรนด์ของ Laptop\n"
        "- **Screen_Size**: ขนาดหน้าจอของ Laptop\n"
        "- **RAM**: ขนาดของ RAM\n"
        "- **Storage**: ความจุของ Storage\n"
        "- **Price**: ราคาของ Laptop\n"
        "- **Rating**: การให้คะแนนจากผู้ใช้"
    )

    # อธิบายความไม่สมบูรณ์ของ Dataset
    st.header("ที่มาของ DDoS Traffic Dataset")
    st.markdown(
        "ข้อมูลที่ใช้ในโปรเจกต์นี้เพิ่มเติมจาก Kaggle: "
        "[DDoS Traffic Dataset](https://www.kaggle.com/datasets/oktayrdeki/ddos-traffic-dataset)\n"
        "Dataset นี้ประกอบด้วยข้อมูลการจราจรเครือข่ายที่ใช้ในการวิเคราะห์การโจมตีแบบ DDoS (Distributed Denial of Service)"
    )
    
    st.write("### ข้อมูลตัวอย่างจาก DDoS Dataset")
<<<<<<< HEAD
    ddos_data = pd.read_csv(r"paokungz/IS_Project/blob/main/data/DDoS_Dataset_with_Missing_Values.csv")
=======
    ddos_data = pd.read_csv(dpath)
>>>>>>> eed7309 (Save changes before rebase)
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
