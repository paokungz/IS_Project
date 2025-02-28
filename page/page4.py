import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, r2_score

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    # Load data
    car_data = pd.read_csv(file_path)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')  # Replace missing values with mean
    car_data[['Selling_Price', 'Present_Price', 'Kms_Driven']] = imputer.fit_transform(
        car_data[['Selling_Price', 'Present_Price', 'Kms_Driven']]
    )

    # Label encoding for categorical columns
    label_encoder = LabelEncoder()
    car_data['Fuel_Type'] = label_encoder.fit_transform(car_data['Fuel_Type'])
    car_data['Seller_Type'] = label_encoder.fit_transform(car_data['Seller_Type'])
    car_data['Transmission'] = label_encoder.fit_transform(car_data['Transmission'])

    return car_data

# Function to train regression models
def train_regression_models(car_data):
    X = car_data.drop(columns=['Car_Name', 'Owner', 'Selling_Price'])  # Features
    y = car_data['Selling_Price']  # Target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    mse_knn = mean_squared_error(y_test, y_pred_knn)

    # Train Decision Tree Regressor
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    mse_dt = mean_squared_error(y_test, y_pred_dt)

    # R² Score
    r2_knn = r2_score(y_test, y_pred_knn)
    r2_dt = r2_score(y_test, y_pred_dt)

    return mse_knn, mse_dt, r2_knn, r2_dt, y_pred_knn, y_pred_dt, y_test, knn, dt

# Function to train classification models
def train_classification_models(car_data):
    threshold = car_data['Selling_Price'].median()
    car_data['Price_Category'] = (car_data['Selling_Price'] > threshold).astype(int)

    X = car_data.drop(columns=['Car_Name', 'Owner', 'Selling_Price', 'Price_Category'])  # Features
    y = car_data['Price_Category']  # Target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    # Train Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    # Precision, Recall, F1 Scores
    precision_knn = precision_score(y_test, y_pred_knn)
    recall_knn = recall_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn)

    precision_dt = precision_score(y_test, y_pred_dt)
    recall_dt = recall_score(y_test, y_pred_dt)
    f1_dt = f1_score(y_test, y_pred_dt)

    return precision_knn, recall_knn, f1_knn, precision_dt, recall_dt, f1_dt, knn, dt

# Main Streamlit app
def app():
    st.title("Car Price Prediction and Classification")

    # Upload data file
    file_path = st.text_input("Enter the file path for the dataset (CSV format)", r"C:\Users\s6404062620087\Desktop\IS_project\data\Car_Data_with_Missing_Values.csv")
    
    # Load and preprocess data
    car_data = load_and_preprocess_data(file_path)

    # Model selection
    model_type = st.selectbox("Select the model type", ["Regression", "Classification"])

    # Input features for prediction
    st.write("### Enter the car details for prediction:")

    # Input fields for prediction
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    present_price = st.number_input("Present Price of the car (in thousands)", min_value=0.0)
    kms_driven = st.number_input("Kms Driven", min_value=0)
    
    # Features for prediction
    features = {
        'Fuel_Type': fuel_type,
        'Seller_Type': seller_type,
        'Transmission': transmission,
        'Present_Price': present_price,
        'Kms_Driven': kms_driven
    }

    # Convert inputs to appropriate format for prediction
    input_data = pd.DataFrame([features])

    # Map categorical inputs to numeric values
    input_data['Fuel_Type'] = input_data['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
    input_data['Seller_Type'] = input_data['Seller_Type'].map({'Individual': 0, 'Dealer': 1})
    input_data['Transmission'] = input_data['Transmission'].map({'Manual': 0, 'Automatic': 1})

    # Prediction and output
    if model_type == "Regression":
        # Train regression models
        mse_knn, mse_dt, r2_knn, r2_dt, y_pred_knn, y_pred_dt, y_test, knn_model, dt_model = train_regression_models(car_data)

        st.write(f'KNN Mean Squared Error: {mse_knn:.2f}')
        st.write(f'Decision Tree Mean Squared Error: {mse_dt:.2f}')

        # Predict the selling price using trained model (KNN)
        predicted_price_knn = knn_model.predict(input_data)
        st.write(f"Predicted Selling Price (KNN): {predicted_price_knn[0]:.2f} thousand")

        # Predict the selling price using trained model (Decision Tree)
        predicted_price_dt = dt_model.predict(input_data)
        st.write(f"Predicted Selling Price (Decision Tree): {predicted_price_dt[0]:.2f} thousand")

    elif model_type == "Classification":
        # Train classification models
        precision_knn, recall_knn, f1_knn, precision_dt, recall_dt, f1_dt, knn_model, dt_model = train_classification_models(car_data)

        st.write(f'KNN Precision: {precision_knn:.2f}')
        st.write(f'Decision Tree Precision: {precision_dt:.2f}')

        # Predict price category using trained model (KNN)
        predicted_category_knn = knn_model.predict(input_data)
        st.write(f"Predicted Price Category (KNN): {'High' if predicted_category_knn[0] == 1 else 'Low'}")

        # Predict price category using trained model (Decision Tree)
        predicted_category_dt = dt_model.predict(input_data)
        st.write(f"Predicted Price Category (Decision Tree): {'High' if predicted_category_dt[0] == 1 else 'Low'}")
