import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data():
    labtopdata = pd.read_csv(r'paokungz/IS_Project/blob/main/data/laptop_prices_mod.csv.csv')
    ddos_data = pd.read_csv(r'paokungz/IS_Project/blob/main/data/DDoS_Dataset_with_Missing_Values.csv')
    return labtopdata, ddos_data

labtopdata, ddos_data = load_data()

def app():
    # แสดงข้อมูลเบื้องต้นเกี่ยวกับการจัดการข้อมูลใน laptop_prices_mod.csv
    st.write("### การจัดการข้อมูลใน laptop_prices_mod.csv:")
    
    st.write("""
    ในขั้นตอนแรก เราจะทำการจัดการกับข้อมูลในไฟล์ `laptop_prices_mod.csv` ซึ่งประกอบไปด้วยข้อมูลเกี่ยวกับราคาคอมพิวเตอร์โน้ตบุ๊กและคุณสมบัติต่าง ๆ ที่เกี่ยวข้อง เช่น RAM, ขนาดหน้าจอ, ความจุของฮาร์ดดิสก์ เป็นต้น โดยมีขั้นตอนหลัก ๆ ดังนี้:
    
    1. **การจัดการค่าที่หายไป (Missing Values)**:
       - เราจะเติมค่าที่หายไปในคอลัมน์ที่เป็นตัวเลข เช่น RAM, ความจุของฮาร์ดดิสก์ และราคาโดยใช้ค่ามัธยฐาน (Median)
    
    2. **การแปลงข้อมูลประเภทข้อความ (Categorical Data)**:
       - การแปลงข้อมูลที่เป็นประเภทข้อความ เช่น ประเภทของ GPU หรือประเภทของหน้าจอ ให้เป็นข้อมูลเชิงตัวเลข (One-Hot Encoding)
    
    3. **การแบ่งข้อมูล**:
       - ข้อมูลจะถูกแบ่งออกเป็นชุดฝึกสอน (Training) และชุดทดสอบ (Testing) เพื่อใช้ในการฝึกโมเดล
    
    4. **การทำนาย**:
       - เราจะใช้โมเดลต่าง ๆ เช่น Random Forest และ Gradient Boosting ในการทำนายราคาของคอมพิวเตอร์จากข้อมูลที่มี
    """)

    # แสดงโค้ดที่ใช้ในการจัดการข้อมูลใน laptop_prices_mod.csv
    st.write("#### โค้ดที่ใช้ในการจัดการข้อมูลใน `laptop_prices_mod.csv`:")

    st.write("##### 1. การโหลดข้อมูล:")
    st.code("""
    # Step 1: การโหลดข้อมูล
    file_path = '/content/laptop_prices_mod.csv'  # พาธไฟล์ที่โหลดขึ้น
    data = pd.read_csv(file_path)
    """)

    st.write("""
    ในขั้นตอนนี้ เราจะทำการโหลดข้อมูลจากไฟล์ CSV ด้วยคำสั่ง `pd.read_csv()` ซึ่งจะโหลดข้อมูลที่เกี่ยวกับคอมพิวเตอร์โน้ตบุ๊กมาเป็น DataFrame ในตัวแปร `data`
    """)

    st.write("##### 2. การเติมค่าที่หายไป (Missing Values):")
    st.code("""
    # Step 2: การจัดการข้อมูล
    # เติมค่าที่หายไป (Missing Values) ในคอลัมน์ตัวเลขด้วยค่ามัธยฐาน
    imputer = SimpleImputer(strategy='median')
    data[['RAM (GB)', 'Storage_Capacity', 'Price_THB']] = imputer.fit_transform(data[['RAM (GB)', 'Storage_Capacity', 'Price_THB']])
    """)

    st.write("""
    ในขั้นตอนนี้ เราจะใช้ `SimpleImputer` เพื่อเติมค่าที่หายไปในคอลัมน์ที่เป็นข้อมูลตัวเลข เช่น **RAM (GB)**, **Storage_Capacity** และ **Price_THB** โดยใช้ค่ามัธยฐาน (Median) ของแต่ละคอลัมน์
    """)

    st.write("##### 3. การแปลงข้อมูลประเภทข้อความ (Categorical Data) โดยใช้ One-Hot Encoding:")
    st.code("""
    # Step 3: One-Hot Encoding สำหรับคอลัมน์ที่เป็น string (Categorical Data)
    data = pd.get_dummies(data, drop_first=True)
    """)

    st.write("""
    ในขั้นตอนนี้ เราจะใช้ `pd.get_dummies()` เพื่อแปลงคอลัมน์ที่เป็นประเภทข้อความ (เช่น **Storage Type**, **Resolution Type**, และ **GPU Model**) ให้เป็นข้อมูลเชิงตัวเลขที่สามารถใช้งานในโมเดล Machine Learning ได้
    """)

    st.write("##### 4. การเตรียมข้อมูลสำหรับโมเดล:")
    st.code("""
    # Step 4: เตรียมข้อมูลสำหรับโมเดล
    X = data.drop(['Price_THB'], axis=1)  # Features
    y = data['Price_THB']  # Target variable (Price in THB)
    """)

    st.write("""
    เราจะเตรียมข้อมูลโดยการแยกข้อมูลที่เป็น Features (ตัวแปรที่ใช้ในการทำนาย) ออกจาก Target (ราคาคอมพิวเตอร์ **Price_THB**) ซึ่งจะถูกเก็บไว้ในตัวแปร `X` และ `y`
    """)

    st.write("##### 5. การเติมค่าที่หายไปใน Features (X):")
    st.code("""
    # Step 5: เติมค่าที่หายไปใน Features (X)
    X_imputed = imputer.fit_transform(X)  # ใช้ SimpleImputer เติมค่าที่หายไปใน Features
    """)

    st.write("""
    ในขั้นตอนนี้ เราจะใช้ `SimpleImputer` เพื่อเติมค่าที่หายไปในข้อมูล Features ที่เหลืออยู่ใน **X** เช่นเดียวกับที่ทำในขั้นตอนก่อนหน้า
    """)

    st.write("##### 6. การแบ่งข้อมูลเป็น Train และ Test:")
    st.code("""
    # Step 6: แบ่งข้อมูลเป็น train และ test
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    """)

    st.write("""
    เราจะใช้ `train_test_split()` เพื่อแบ่งข้อมูลออกเป็นสองชุดคือ **train** และ **test** เพื่อใช้ฝึกโมเดลและทดสอบโมเดลที่ฝึกแล้ว
    """)

    st.write("##### 7. การสร้างและฝึกโมเดล Random Forest:")
    st.code("""
    # Step 7: สร้างและฝึกโมเดล Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    """)

    st.write("""
    ในขั้นตอนนี้ เราจะใช้โมเดล **Random Forest** ในการฝึกจากข้อมูลที่แบ่งเป็นชุดฝึก (`X_train`, `y_train`) โดยใช้ 100 ต้นไม้ใน Random Forest
    """)

    st.write("##### 8. การสร้างและฝึกโมเดล Gradient Boosting:")
    st.code("""
    # Step 8: สร้างและฝึกโมเดล Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    """)

    st.write("""
    เช่นเดียวกับขั้นตอนที่แล้ว เราจะใช้โมเดล **Gradient Boosting** ในการฝึกจากชุดข้อมูลฝึก (`X_train`, `y_train`)
    """)

    st.write("##### 9. การทำนายด้วยโมเดล Random Forest และ Gradient Boosting:")
    st.code("""
    # Step 9: ตัวอย่างข้อมูลใหม่เพื่อทำการทำนาย
    new_data = {
        'RAM (GB)': [16],  # Example: 16 GB RAM
        'Screen Size (inch)': [15.6],  # Example: 15.6 inch screen
        'Battery Life (hours)': [10],  # Example: 10 hours of battery life
        'Weight (kg)': [1.8],  # Example: 1.8 kg weight
        'Storage_Capacity': [512],  # Example: 512 GB storage
        'Storage_Type_SSD': [1],  # Example: SSD (One-Hot Encoding for Storage_Type)
        'Resolution_Type_Full HD': [1],  # Example: Full HD (One-Hot Encoding for Resolution_Type)
        'GPU_Model_Nvidia GTX 1650': [1],  # Example: Nvidia GTX 1650 (One-Hot Encoding for GPU_Model)
    }

    # แปลงตัวอย่างข้อมูลใหม่เป็น DataFrame
    new_data_df = pd.DataFrame(new_data)

    # ใช้ columns ที่เหมือนกับ training data โดยการเติมคอลัมน์ที่หายไป
    missing_cols = set(X.columns) - set(new_data_df.columns)
    for col in missing_cols:
        new_data_df[col] = 0  # เติมคอลัมน์ที่หายไปด้วยค่า 0
    new_data_df = new_data_df[X.columns]  # จัดลำดับคอลัมน์ให้ตรงกับ X_train

    # เติมค่าที่หายไปในตัวอย่างข้อมูลใหม่
    new_data_imputed = imputer.transform(new_data_df)

    # ทำนายราคาโดยใช้ Random Forest
    rf_pred = rf_model.predict(new_data_imputed)
    print("Predicted Price by Random Forest:", rf_pred)

    # ทำนายราคาโดยใช้ Gradient Boosting
    gb_pred = gb_model.predict(new_data_imputed)
    print("Predicted Price by Gradient Boosting:", gb_pred)
    """)

    st.write("""
    ในขั้นตอนนี้ เราจะใช้โมเดลทั้งสอง ได้แก่ **Random Forest** และ **Gradient Boosting** ในการทำนายราคาของคอมพิวเตอร์โน้ตบุ๊กจากข้อมูลใหม่ที่ให้มา
    """)

    # การจัดการข้อมูลใน DDoS Dataset
    st.write("### การจัดการข้อมูลใน DDoS Dataset:")

    st.write("""
    ในส่วนนี้เราจะทำการจัดการข้อมูลใน `DDoS_Dataset_with_Missing_Values.csv` ซึ่งเป็นข้อมูลเกี่ยวกับการโจมตีแบบ DDoS โดยข้อมูลจะประกอบไปด้วยค่าต่าง ๆ เช่น พอร์ตแหล่งที่มา, พอร์ตปลายทาง, ขนาดของแพ็กเกจ และข้อมูลการโจมตีอื่น ๆ
    """)

    # แสดงโค้ดที่ใช้ในการจัดการข้อมูลใน DDoS
    st.write("#### โค้ดที่ใช้ในการจัดการข้อมูลใน DDoS Dataset:")
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

    st.write("""
    ในส่วนนี้ เราใช้ `SimpleImputer` เติมค่าที่หายไปในคอลัมน์ที่เป็นข้อมูลตัวเลข และใช้ `LabelEncoder` เพื่อแปลงข้อมูลประเภทข้อความในคอลัมน์ **`target`** ให้เป็นตัวเลข
    """)

    # สรุปการจัดการข้อมูลใน laptop_prices_mod.csv
    st.write("### สรุปการจัดการข้อมูลใน `laptop_prices_mod.csv`:")
    st.write("""
    1. **การเติมค่าที่หายไป (Missing Values)**: ค่าที่หายไปในคอลัมน์ **RAM (GB)**, **Storage_Capacity**, และ **Price_THB** ได้ถูกเติมด้วย **ค่ามัธยฐาน** (Median)
    2. **การแปลงข้อมูลประเภทข้อความ**: คอลัมน์ที่เป็นข้อความ เช่น **Storage Type**, **Resolution Type**, และ **GPU Model** ถูกแปลงเป็นข้อมูลเชิงตัวเลขด้วย **One-Hot Encoding**
    3. **การแบ่งข้อมูล**: ข้อมูลถูกแบ่งเป็น **train** และ **test** สำหรับการฝึกโมเดล
    4. **โมเดลที่ใช้**: ใช้โมเดล **Random Forest** และ **Gradient Boosting** ในการทำนายราคาคอมพิวเตอร์โน้ตบุ๊กจากคุณสมบัติ
    """)

    # สรุปการจัดการข้อมูลใน DDoS Dataset
    st.write("### สรุปการจัดการข้อมูลใน `DDoS_Dataset_with_Missing_Values.csv`:")
    st.write("""
    1. **การเติมค่าที่หายไป (Missing Values)**: ค่าที่หายไปในคอลัมน์ที่เป็นข้อมูลตัวเลขได้ถูกเติมด้วย **ค่าเฉลี่ย** (Mean)
    2. **การแปลงข้อมูลประเภทข้อความ**: คอลัมน์ **`target`** ถูกแปลงเป็น **ตัวเลข** โดยใช้ **LabelEncoder**
    3. **การแสดงผล**: ข้อมูลที่เติมค่าหายไปแล้วถูกแสดงให้เห็น
    """)

