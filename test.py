import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from transformers import AutoModelForCausalLM, AutoTokenizer
from cryptography.fernet import Fernet
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Define your Hugging Face API key here
apikey = "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"

# Initialize the tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Encryption key and cipher
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Helper Functions for Text Generation, Encryption, and Decryption
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
        return decrypted_data
    except Exception as e:
        st.error(f"Decryption failed: {e}")
        raise

# Title
st.title('AI Assistant for Data Science 🤖')

# Sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with a CSV File.*')
    st.caption('Upload a CSV file to start.')
    st.divider()
    st.caption("<p style='text-align:center'> made with ❤️ by Team Avengers</p>", unsafe_allow_html=True)

# Function to handle missing values with options for imputation
def handle_missing_values(df, method='mean'):
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'mode':
        imputer = SimpleImputer(strategy='most_frequent')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)

    # Apply imputation to numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    imputed_df = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
    df[numeric_df.columns] = imputed_df
    return df

# Outlier Detection and Removal Function
def handle_outliers(df, factor=1.5):
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Plotly Visualizations for Interactive EDA
def plot_missing_values_heatmap(df):
    fig = px.imshow(df.isnull(), title='Missing Values Heatmap')
    st.plotly_chart(fig)

def plot_histograms(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        fig = px.histogram(df, x=col, nbins=30, title=f'Histogram of {col}')
        st.plotly_chart(fig)

def plot_pairplot(df):
    num_df = df.select_dtypes(include=[np.number])
    if len(num_df.columns) > 1:
        fig = ff.create_scatterplotmatrix(num_df, height=800, width=800, diag='histogram')
        st.plotly_chart(fig)

# File Upload Section
user_csv = st.file_uploader("Upload your file here", type="csv")
if user_csv:
    try:
        df = pd.read_csv(user_csv)
    except Exception as e:
        st.error(f"Failed to read the CSV file: {e}")
        st.stop()

    # Missing Values Handling
    st.sidebar.subheader("Handle Missing Values")
    imputation_method = st.sidebar.selectbox("Choose Imputation Method", ["mean", "median", "mode", "knn"])
    df = handle_missing_values(df, method=imputation_method)

    # Outlier Handling
    st.sidebar.subheader("Outlier Handling")
    if st.sidebar.checkbox("Remove Outliers", value=True):
        df = handle_outliers(df, factor=1.5)

    # Feature Engineering - Encoding Categorical Variables
    st.sidebar.subheader("Feature Encoding")
    encode_features = st.sidebar.multiselect("Select Categorical Features to Encode", df.select_dtypes(include=['object']).columns)
    if encode_features:
        encoder = OneHotEncoder(sparse=False)
        encoded_cols = pd.DataFrame(encoder.fit_transform(df[encode_features]), columns=encoder.get_feature_names_out(encode_features))
        df = pd.concat([df.drop(encode_features, axis=1), encoded_cols], axis=1)

    # Display Processed Data and Basic EDA
    st.write("### Data Overview")
    st.write(df.head())

    st.write("### Missing Values Heatmap")
    plot_missing_values_heatmap(df)

    st.write("### Histograms of Numerical Features")
    plot_histograms(df)

    st.write("### Pair Plot of Numerical Features")
    plot_pairplot(df)

    # Display DataFrame statistics
    st.write("### Data Summary")
    st.write(df.describe())

    # Original Model Training Code (unchanged)
    st.header('Predictive Modeling')
    feature_cols = st.multiselect("Select Feature Columns", df.columns.tolist())
    target_col = st.selectbox("Select Target Column", df.columns.tolist())
    num_years = st.number_input("Enter number of years for prediction", min_value=1, value=1)
    model_choice = st.selectbox("Select Machine Learning Model", ["Linear Regression", "Random Forest", "Gradient Boosting", "Support Vector Regression"])

    if st.button("Train Model"):
        if target_col not in df.columns or not set(feature_cols).issubset(df.columns):
            st.error("Please select valid feature and target columns.")
        else:
            df = df.dropna(subset=[target_col] + feature_cols)
            X = df[feature_cols]
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = {"Linear Regression": LinearRegression(),
                     "Random Forest": RandomForestRegressor(),
                     "Gradient Boosting": GradientBoostingRegressor(),
                     "Support Vector Regression": SVR()}[model_choice]

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            rmse = mse ** 0.5
            st.write(f"**Model Accuracy (RMSE):** {rmse:.2f}")

            future_prediction = model.predict(pd.DataFrame(X_test.iloc[-num_years:]))
            trend = "high" if future_prediction.mean() > y.mean() else "low"
            st.write(f"The market is expected to be {trend} in the future after {num_years} years.")
            st.write("**Predictions:**")
            st.write(predictions)
