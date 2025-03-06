# 1_main.py
import streamlit as st
from pathlib import Path
from page1 import app1
from page2 import app2
from page3 import app3
from page4 import app4
from page5 import app5

def main():
    # เมนูเลือกหน้า
    page = st.selectbox("เลือกหน้าที่ต้องการ", ["การเตรียมข้อมูล", "ทฤษฎีของอัลกอริทึมที่พัฒนา","การจัดการข้อมูล"," Random Forest และ Gradient Boosting","Neural Network"])

    if page == "การเตรียมข้อมูล":
        app1()  # เรียกใช้ฟังก์ชัน app() จาก page1
    elif page == "ทฤษฎีของอัลกอริทึมที่พัฒนา":
        app2()
    elif page == "การจัดการข้อมูล":
        app3()
    elif page == " Random Forest และ Gradient Boosting":
        app4()
    elif page == "Neural Network":
        app5()
if __name__ == "__main__":
    main()
