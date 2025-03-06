import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ฟังก์ชันหลักของแอพ
def app():
    
    # Function to load a saved model
    def load_model(model_path):
        return joblib.load(model_path)

    # Function to predict the price using the provided model
    def predict_price(model, input_data):
        return model.predict(pd.DataFrame([input_data]))[0]
    # Function to collect user input
    def get_user_input():

        st.sidebar.header("กรุณากรอกข้อมูล Laptop")

        # Step 1: กรอกข้อมูล RAM
        ram = st.sidebar.number_input("RAM (GB)", min_value=1, max_value=64, value=8, help="เลือกขนาดของ RAM ที่คุณต้องการ เช่น 8GB, 16GB")
        
        # Step 2: กรอกข้อมูล Storage
        storage = st.sidebar.number_input("Storage (GB)", min_value=120, max_value=2000, value=512, help="เลือกขนาดของ Storage เช่น 512GB, 1TB")
        
        # Step 3: กรอกข้อมูลขนาดหน้าจอ (Screen Size)
        screen_size = st.sidebar.number_input("Screen Size (inch)", min_value=10, max_value=17, value=15, help="เลือกขนาดหน้าจอของ Laptop เช่น 15.6 นิ้ว")
        
        # Step 4: กรอกข้อมูล Battery Life
        battery_life = st.sidebar.number_input("Battery Life (hours)", min_value=1, max_value=20, value=10, help="เลือกระยะเวลาใช้งานแบตเตอรี่ในชั่วโมง เช่น 10 ชั่วโมง")
        
        # Step 5: กรอกข้อมูลน้ำหนัก (Weight)
        weight = st.sidebar.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.8, help="เลือกน้ำหนักของ Laptop เช่น 1.8 kg")
        
        # Step 6: เลือก GPU
        gpu = st.sidebar.selectbox("GPU Model", ['RTX 2060', 'RTX 3060', 'RTX 3080', 'RX 6600', 'RX 6800'], help="เลือก GPU ที่มีใน Laptop")
        
        # Step 7: เลือก Processor
        processor = st.sidebar.selectbox("Processor", ['AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9', 'Intel i3', 'Intel i5', 'Intel i7', 'Intel i9'], help="เลือก Processor ที่มีใน Laptop")
        
        # Step 8: เลือกระบบปฏิบัติการ
        os = st.sidebar.selectbox("Operating System", ['Linux', 'Windows', 'macOS'], help="เลือกระบบปฏิบัติการของ Laptop")
        
        # Step 9: เลือกประเภทของ Storage
        storage_type = st.sidebar.selectbox("Storage Type", ['SSD'], help="เลือกประเภทของ Storage เช่น SSD")
        
        # Step 10: เลือกประเภทของ Resolution
        resolution_type = st.sidebar.selectbox("Resolution Type", ['4K', 'Full HD', 'HD', 'Other'], help="เลือกประเภทของ Resolution")
        
        # Step 11: เลือก Brand
        brand = st.sidebar.selectbox("Brand", ['Apple', 'Asus', 'Dell', 'HP', 'Lenovo', 'MSI', 'Microsoft', 'Razer', 'Samsung'], help="เลือก Brand ของ Laptop")

        # One-Hot Encoding for categorical variables
        input_data = {
            'RAM (GB)': ram,
            'Storage_Capacity': storage,
            'Screen Size (inch)': screen_size,
            'Battery Life (hours)': battery_life,
            'Weight (kg)': weight,
            'GPU_Model_' + gpu: 1,
            'Processor_' + processor: 1,
            'Operating System_' + os: 1,
            'Storage_Type_' + storage_type: 1,
            'Resolution_Type_' + resolution_type: 1,
            'Brand_' + brand: 1
        }
        
        # Ensure all possible columns are present, adding missing ones with 0s
        all_columns = ['RAM (GB)', 'Storage_Capacity', 'Screen Size (inch)', 'Battery Life (hours)', 'Weight (kg)'] + \
                       ['GPU_Model_RTX 2060', 'GPU_Model_RTX 3060', 'GPU_Model_RTX 3080', 'GPU_Model_RX 6600', 'GPU_Model_RX 6800'] + \
                       ['Processor_AMD Ryzen 5', 'Processor_AMD Ryzen 7', 'Processor_AMD Ryzen 9', 'Processor_Intel i3', 'Processor_Intel i5', 
                        'Processor_Intel i7', 'Processor_Intel i9'] + \
                       ['Operating System_Linux', 'Operating System_Windows', 'Operating System_macOS'] + \
                       ['Storage_Type_SSD'] + \
                       ['Resolution_Type_4K', 'Resolution_Type_Full HD', 'Resolution_Type_HD', 'Resolution_Type_Other'] + \
                       ['Brand_Apple', 'Brand_Asus', 'Brand_Dell', 'Brand_HP', 'Brand_Lenovo', 'Brand_MSI', 'Brand_Microsoft', 'Brand_Razer', 'Brand_Samsung']
        
        # Fill missing columns with 0
        for col in all_columns:
            if col not in input_data:
                input_data[col] = 0
        
        return input_data

    # โหลดโมเดล RandomForest และ GradientBoosting
    rf_model = load_model('paokungz/IS_Project/blob/main/page/rf_model.pkl')
    gb_model = load_model('paokungz/IS_Project/blob/main/page/gb_model.pkl')

    # แสดงเนื้อหาภายในแอพ
    st.title("ทำนายราคา Laptop")
    st.write("กรุณากรอกข้อมูลที่คุณต้องการทำนายราคา แล้วคลิกปุ่ม 'ทำนายราคา'")

    # รับข้อมูลจากผู้ใช้
    input_data = get_user_input()

    # ปุ่มให้ทำนายราคา
    if st.sidebar.button('ทำนายราคา'):
        # ทำนายราคาด้วยโมเดล Random Forest และ Gradient Boosting
        rf_price = predict_price(rf_model, input_data)
        gb_price = predict_price(gb_model, input_data)
        
        # แสดงผลลัพธ์ในขนาดใหญ่
        st.markdown(f"<h2 style='color: blue;'>ราคาที่ทำนายจากโมเดล Random Forest: {rf_price:,.2f} THB</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: green;'>ราคาที่ทำนายจากโมเดล Gradient Boosting: {gb_price:,.2f} THB</h2>", unsafe_allow_html=True)

    # แสดงโค้ดให้ผู้ใช้งาน
    st.subheader("โค้ดสำหรับแอพนี้")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    # Step 1: การโหลดข้อมูล
    st.markdown("""
    ### Step 1: การโหลดข้อมูล
    ในขั้นตอนแรกเราทำการโหลดข้อมูลจากไฟล์ CSV โดยใช้ `pandas.read_csv` เพื่อให้ข้อมูลพร้อมสำหรับการวิเคราะห์
    """)
    st.code("""
    file_path = '/content/laptop_prices_mod.csv'  # พาธไฟล์ที่โหลดขึ้น
    data = pd.read_csv(file_path)
    """, language="python")

    # Step 2: การจัดการข้อมูล
    st.markdown("""
    ### Step 2: การจัดการข้อมูล
    ในขั้นตอนนี้เราจะเติมค่าที่หายไปในคอลัมน์ตัวเลขด้วยค่ามัธยฐาน โดยใช้ `SimpleImputer` ของ `sklearn`
    """)
    st.code("""
    # เติมค่าที่หายไป (Missing Values) ในคอลัมน์ตัวเลขด้วยค่ามัธยฐาน
    imputer = SimpleImputer(strategy='median')
    data[['RAM (GB)', 'Storage_Capacity', 'Price_THB']] = imputer.fit_transform(data[['RAM (GB)', 'Storage_Capacity', 'Price_THB']])
    """, language="python")

    # Step 3: One-Hot Encoding สำหรับคอลัมน์ที่เป็น string (Categorical Data)
    st.markdown("""
    ### Step 3: One-Hot Encoding สำหรับคอลัมน์ที่เป็น string (Categorical Data)
    เนื่องจากข้อมูลที่เป็นข้อความไม่สามารถใช้ในการฝึกโมเดลได้ เราจึงต้องใช้ One-Hot Encoding เพื่อแปลงข้อมูลประเภทนี้ให้เป็นตัวเลข
    """)
    st.code("""
    # One-Hot Encoding สำหรับคอลัมน์ที่เป็น string (Categorical Data)
    data = pd.get_dummies(data, drop_first=True)
    """, language="python")

    # Step 4: เตรียมข้อมูลสำหรับโมเดล
    st.markdown("""
    ### Step 4: เตรียมข้อมูลสำหรับโมเดล
    หลังจากที่แปลงข้อมูลแล้ว เราจะเตรียมข้อมูล (Features) และเป้าหมาย (Target) สำหรับการฝึกโมเดล โดยจะนำคอลัมน์ `Price_THB` เป็น Target
    """)
    st.code("""
    X = data.drop(['Price_THB'], axis=1)  # Features
    y = data['Price_THB']  # Target variable (Price in THB)
    """, language="python")

    # Step 5: เติมค่าที่หายไปใน Features (X)
    st.markdown("""
    ### Step 5: เติมค่าที่หายไปใน Features (X)
    ในขั้นตอนนี้เราจะเติมค่าที่หายไปใน Features โดยใช้ `SimpleImputer` เพื่อลบค่าที่หายไปใน Features ทั้งหมด
    """)
    st.code("""
    # เติมค่าที่หายไปใน Features (X)
    X_imputed = imputer.fit_transform(X)  # ใช้ SimpleImputer เติมค่าที่หายไปใน Features
    """, language="python")

    # Step 6: แบ่งข้อมูลเป็น train และ test
    st.markdown("""
    ### Step 6: แบ่งข้อมูลเป็น train และ test
    การแบ่งข้อมูลออกเป็นชุดฝึก (Train Set) และชุดทดสอบ (Test Set) ช่วยให้เราสามารถฝึกโมเดลและทดสอบความแม่นยำได้
    """)
    st.code("""
    # แบ่งข้อมูลเป็น train และ test
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    """, language="python")

    # Step 7: สร้างและฝึกโมเดล Random Forest
    st.markdown("""
    ### Step 7: สร้างและฝึกโมเดล Random Forest
    เราใช้ RandomForestRegressor สำหรับการฝึกโมเดลทำนายราคาลาปท็อป โดยจะใช้ข้อมูลที่แยกออกเป็นชุดฝึก
    """)
    st.code("""
    # สร้างและฝึกโมเดล Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    """, language="python")

    # Step 8: สร้างและฝึกโมเดล Gradient Boosting
    st.markdown("""
    ### Step 8: สร้างและฝึกโมเดล Gradient Boosting
    ในขั้นตอนนี้เราจะสร้างโมเดล Gradient Boosting สำหรับทำนายราคาลาปท็อปโดยการฝึกโมเดลนี้ด้วยชุดข้อมูลที่เตรียมไว้
    """)
    st.code("""
    # สร้างและฝึกโมเดล Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    """, language="python")

    # Step 9: Visualization - แสดงการทำนายของทั้งสองโมเดล
    st.markdown("""
    ### Step 9: Visualization - แสดงการทำนายของทั้งสองโมเดล
    เราจะสร้างกราฟ Scatter Plot เพื่อเปรียบเทียบค่าที่ทำนายได้จากทั้งสองโมเดล (Random Forest และ Gradient Boosting)
    """)
    st.code("""
    # กราฟการทำนาย Random Forest
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_test, y=y_pred_rf, color='blue', label='Random Forest')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title('Random Forest: True vs Predicted Price')
    plt.legend()
    plt.show()

    # กราฟการทำนาย Gradient Boosting
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_test, y=y_pred_gb, color='green', label='Gradient Boosting')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title('Gradient Boosting: True vs Predicted Price')
    plt.legend()
    plt.show()
    """, language="python")
    st.image(r'paokungz/IS_Project/blob/main/page/page4.png', caption="random forest", use_container_width=True, output_format='auto')
    st.image(r'paokungz/IS_Project/blob/main/page/page4_2.png.png', caption="Gradient boost", use_container_width=True, output_format='auto')
    st.markdown("""
    ### แสดงการทำนายของทั้งสองโมเดล
    Random Forest Mean Squared Error: 188917136.84570354\n
    Random Forest R²: 0.912973305902608\n
    Gradient Boosting Mean Squared Error: 181326597.95363292\n
    Gradient Boosting R²: 0.916469968604701
    """)
