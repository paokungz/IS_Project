import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def app():
    # หัวข้อหน้า
    st.title("🧠 ทฤษฎีของอัลกอริธึม Neural Network, Random Forest และ Gradient Boosting")

    # อธิบายเกี่ยวกับ Neural Network
    st.write("""
    ## 1️⃣ Neural Network (โครงข่ายประสาทเทียม)
    Neural Networks (NN) เป็นอัลกอริธึมที่เลียนแบบการทำงานของสมองมนุษย์ โดยประกอบด้วยหน่วยประมวลผลที่เรียกว่า **Neuron** ซึ่งเชื่อมโยงกันในรูปแบบของชั้นต่าง ๆ ได้แก่:
    - **Input Layer**: รับข้อมูลเข้ามา
    - **Hidden Layers**: ประมวลผลข้อมูล
    - **Output Layer**: ให้ผลลัพธ์ออกมา

    **หลักการทำงาน**:

    เมื่อข้อมูลถูกป้อนเข้าสู่ Input Layer ข้อมูลจะถูกส่งผ่าน Hidden Layers ซึ่งแต่ละชั้นจะมีการคำนวณค่าผ่านฟังก์ชันที่เรียกว่า **Activation Function** เช่น **ReLU** (Rectified Linear Unit) หรือ **Sigmoid** เพื่อเพิ่มความไม่เป็นเชิงเส้นให้กับโมเดล จากนั้นผลลัพธ์จะถูกส่งไปยัง Output Layer เพื่อให้ผลลัพธ์สุดท้าย

    **การฝึกสอน (Training)**:

    การฝึกสอน Neural Network ใช้กระบวนการที่เรียกว่า **Backpropagation** ซึ่งเป็นการปรับปรุงน้ำหนัก (Weights) และอคติ (Biases) ของการเชื่อมต่อระหว่าง Neuron โดยอาศัยค่าความผิดพลาด (Error) ที่คำนวณได้จากผลลัพธ์ที่ได้และผลลัพธ์ที่คาดหวัง

    **ประเภทของ Activation Function**:
    - **Sigmoid**: นิยมใช้ในงานที่มีผลลัพธ์เป็นค่าระหว่าง 0 และ 1
    - **ReLU**: ใช้ในกรณีที่ต้องการให้ค่าผลลัพธ์ที่เป็นบวกเท่านั้น
    - **Softmax**: ใช้ในงานประเภท **classification** สำหรับการคำนวณความน่าจะเป็น

    **ข้อดี**:
    - สามารถเรียนรู้จากข้อมูลที่ซับซ้อน
    - ใช้ได้ดีในงาน **Image Recognition**, **Speech Recognition**, **Natural Language Processing** เป็นต้น

    **ข้อเสีย**:
    - ต้องการข้อมูลที่มีคุณภาพและการเตรียมข้อมูลที่ดี
    - ใช้เวลาฝึกนานและมีการคำนวณที่หนัก

    **📚 อ้างอิง**:
    - [Neural Networks - DeepLearning.ai](https://www.deeplearning.ai/)
    - [Neural Networks - Scikit-learn Documentation](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
    - [A Comprehensive Guide to Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-neural-networks-9d0e7c7068bc)
    """)

    # อธิบายเกี่ยวกับ Random Forest
    st.write("""
    ## 2️⃣ Random Forest (ป่าไม้สุ่ม)
    **Random Forest** เป็นอัลกอริธึมที่ใช้ **Ensemble Learning** โดยการสร้างต้นไม้การตัดสินใจหลายๆ ต้น (Decision Trees) เพื่อให้ผลลัพธ์ที่แม่นยำและคาดเดาได้ดีกว่า โดยแต่ละต้นไม้จะถูกฝึกด้วยชุดข้อมูลที่สุ่มมา (Bootstrap Sampling) และจะมีการตัดสินใจจากการลงคะแนนของต้นไม้ทั้งหมด (Voting) หรือค่าเฉลี่ยของผลลัพธ์ (Regression)

    **ลักษณะของ Random Forest**:
    - สร้าง **หลายๆ ต้นไม้** ที่ใช้ในกระบวนการตัดสินใจ
    - การตัดสินใจจาก **การลงคะแนนของหลายๆ ต้นไม้** (ในกรณี Classification) หรือ **ค่าเฉลี่ยของผลลัพธ์** (ในกรณี Regression)

    **หลักการทำงาน**:

    Random Forest ใช้ **หลายๆ ต้นไม้** เพื่อให้การทำนายมีความเสถียรมากขึ้น และลดปัญหาที่เกิดจาก Overfitting ในกรณีที่ใช้ Decision Tree เพียงต้นเดียว

    **ข้อดี**:
    - สามารถทำงานได้ดีแม้ในข้อมูลที่มีการกระจายตัว (non-linear data)
    - สามารถจัดการกับข้อมูลที่มีความซับซ้อน
    - ไม่ค่อยเกิด Overfitting เพราะการรวมผลลัพธ์จากหลายๆ ต้นไม้

    **ข้อเสีย**:
    - การฝึกฝนและการทำนายอาจใช้เวลานาน
    - การแปลผลลัพธ์อาจไม่สะดวกเมื่อเทียบกับ Decision Trees เดี่ยวๆ

    **📚 อ้างอิง**:
    - [Random Forest - Scikit-learn Documentation](https://scikit-learn.org/stable/modules/ensemble.html#random-forest)
    - [Introduction to Random Forests - Towards Data Science](https://towardsdatascience.com/a-comprehensive-introduction-to-random-forest-34f28e6b5c5b)
    """)

    # อธิบายเกี่ยวกับ Gradient Boosting
    st.write("""
    ## 3️⃣ Gradient Boosting (การเสริมกำลังด้วยเกรเดียนต์)
    **Gradient Boosting** เป็นอัลกอริธึมที่ใช้ **Ensemble Learning** เช่นเดียวกับ Random Forest แต่การเสริมกำลังจะทำในลักษณะการฝึกแต่ละโมเดลใหม่ให้ดีขึ้นจากโมเดลก่อนหน้า โดยการปรับค่าของน้ำหนักในแต่ละการคำนวณตามค่าความผิดพลาดที่เกิดขึ้น (Residuals)

    **ลักษณะของ Gradient Boosting**:
    - เริ่มจากโมเดลฐาน (Base Model) เช่น Decision Tree ขนาดเล็ก
    - สร้างโมเดลใหม่โดยการปรับค่าผลลัพธ์จากโมเดลที่มีอยู่ให้ดีขึ้นทีละขั้น
    - ใช้ **Gradient Descent** เพื่อหาค่าความผิดพลาดในแต่ละขั้นตอน

    **หลักการทำงาน**:

    Gradient Boosting ทำการ **เสริมกำลัง** โดยการพัฒนาแต่ละโมเดลให้ดีขึ้นจากโมเดลก่อนหน้า โดยการลดค่าความผิดพลาด (Residuals) ที่เกิดขึ้นจากโมเดลก่อนหน้า

    **ข้อดี**:
    - มีความแม่นยำสูงและสามารถใช้ได้ดีในงาน **Regression** และ **Classification**
    - ไม่ต้องการการเตรียมข้อมูลมาก

    **ข้อเสีย**:
    - อาจเกิด Overfitting หากไม่ได้ควบคุมการฝึกให้ดี
    - การฝึกอาจใช้เวลานานเพราะเป็นกระบวนการที่ค่อยๆ ปรับปรุงโมเดล

    **📚 อ้างอิง**:
    - [Gradient Boosting - Scikit-learn Documentation](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
    - [Understanding Gradient Boosting Machines - Towards Data Science](https://towardsdatascience.com/understanding-gradient-boosting-machines-6d3101e7e5a6)
    """)

    # สรุปเกี่ยวกับ Random Forest และ Gradient Boosting
    st.write("""
    ### **สรุป** 
    - **Random Forest** เหมาะสมกับงานที่ต้องการการทำนายที่มีความเสถียรและลดปัญหาการ Overfitting โดยการใช้หลายต้นไม้ในการตัดสินใจ
    - **Gradient Boosting** เหมาะสมกับงานที่ต้องการการเสริมกำลังทีละขั้นเพื่อให้โมเดลแม่นยำมากขึ้น แต่ต้องระวัง Overfitting หากไม่ควบคุมการฝึกให้ดี

    **ทั้งสองอัลกอริธึมนี้มีการใช้งานที่หลากหลายและสามารถประยุกต์ใช้ในงานต่างๆ ได้**
    """)