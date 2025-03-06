# 1_main.py
import streamlit as st
from page.page1 import app as page1_app  # เรียกใช้ฟังก์ชัน app() จาก page1
from page.page2 import app as page2_app  # เรียกใช้ฟังก์ชัน app() จาก page2
from page.page3 import app as page3_app  # เรียกใช้ฟังก์ชัน app() จาก page3
from page.page4 import app as page4_app  # เรียกใช้ฟังก์ชัน app() จาก page3
from page.page5 import app as page5_app  # เรียกใช้ฟังก์ชัน app() จาก page3


def main():
    # เมนูเลือกหน้า
    page = st.selectbox("เลือกหน้าที่ต้องการ", ["การเตรียมข้อมูล", "ทฤษฎีของอัลกอริทึมที่พัฒนา","การจัดการข้อมูล"," Random Forest และ Gradient Boosting","Neural Network"])

    if page == "การเตรียมข้อมูล":
        page1_app()  # เรียกใช้ฟังก์ชัน app() จาก page1
    elif page == "ทฤษฎีของอัลกอริทึมที่พัฒนา":
        page2_app()
    elif page == "การจัดการข้อมูล":
        page3_app()
    elif page == " Random Forest และ Gradient Boosting":
        page4_app()
    elif page == "Neural Network":
        page5_app()
if __name__ == "__main__":
    main()
