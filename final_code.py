import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from cryptography.fernet import Fernet
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

# Define your Hugging Face API key here
apikey = "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"

# Initialize the tokenizer and model
model_name = "gpt2"  # Replace with the model you need
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize encryption key and cipher
key = Fernet.generate_key()
cipher_suite = Fernet(key)

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

def plt_to_base64(fig):
    """Convert matplotlib figure to base64 for embedding in HTML"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Title
st.title('AI Assistant for Data Science 🤖')

# Welcoming message
st.write("Hello, 👋 I am your AI Assistant, and I am here to help you with your data science projects.")

# Explanation sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with a CSV File.*')
    st.caption('''**Upload a CSV file to begin. I will analyze the data, provide insights, generate visualizations, 
    and suggest appropriate machine learning models to tackle your problem. Let’s dive into your data science journey!**
    ''')
    
    st.divider()
    
    st.caption("<p style ='text-align:center'> made with ❤️ by Team Avengers</p>", unsafe_allow_html=True)

# Initialize the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        # Encrypt the file for demonstration
        file_data = user_csv.read()
        encrypted_file = encrypt_data(file_data.decode(errors='ignore'))

        # Notify user about encryption
        st.write("Your data has been successfully encrypted.")

        try:
            decrypted_file = decrypt_data(encrypted_file)
            df = pd.read_csv(io.StringIO(decrypted_file), low_memory=False)
        except Exception as e:
            st.error(f"An error occurred during decryption: {e}")
            st.stop()

        # EDA Steps Function
        @st.cache_data
        def steps_eda():
            steps_eda = generate_text('What are the steps of EDA?')
            return steps_eda

        # Functions for EDA
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())

            st.write("**Data Cleaning**")
            st.write("Missing values in each column:")
            null_values = df.isnull().sum()
            st.write(null_values[null_values > 0])

            st.write("Duplicate values in the dataset:")
            duplicates = df[df.duplicated()]
            if not duplicates.empty:
                st.write(duplicates)
            else:
                st.write("No duplicate values found.")

            st.write("**Data Summarisation**")
            st.write(df.describe())

            st.write("**Visualizations**")

            # Heatmap of missing values
            fig, ax = plt.subplots()
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
            ax.set_title('Heatmap of Missing Values')
            st.pyplot(fig)

            # Histograms of numeric features
            num_features = df.select_dtypes(include=['number']).columns
            if len(num_features) > 0:
                fig, ax = plt.subplots()
                df[num_features].hist(ax=ax, bins=30, figsize=(10, 7))
                ax.set_title('Histograms of Numeric Features')
                st.pyplot(fig)
            else:
                st.write("No numeric features available for histograms.")

            # Pairplot for visualizing relationships between numeric features
            if len(num_features) > 1:
                pairplot_fig = sns.pairplot(df[num_features])
                pairplot_fig.fig.suptitle('Pairplot of Numeric Features', y=1.02)
                st.pyplot(pairplot_fig.fig)

        @st.cache_data
        def function_question_variable(selected_variable):
            st.write(f"**Analyzing Variable: {selected_variable}**")
            
            # Instructions for users
            st.info("""
            **Instructions:**
            - You will see the summary statistics, distribution, and outliers for the selected variable.
            - These insights help understand the variable's data patterns, outliers, and missing values.
            """)

            # Summary statistics
            summary_statistics = df[selected_variable].describe()
            st.write("**Summary Statistics**")
            st.write(summary_statistics)

            # Distribution plot with improved design
            st.write("**Distribution of the Variable**")
            fig, ax = plt.subplots()
            sns.histplot(df[selected_variable], kde=True, color="skyblue", ax=ax)
            ax.set_title(f'Distribution of {selected_variable}', fontsize=14, weight='bold')
            ax.set_xlabel(selected_variable, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            st.pyplot(fig)

            # Outlier detection
            st.write("**Outlier Detection**")
            lower_bound = df[selected_variable].quantile(0.25) - 1.5 * (df[selected_variable].quantile(0.75) - df[selected_variable].quantile(0.25))
            upper_bound = df[selected_variable].quantile(0.75) + 1.5 * (df[selected_variable].quantile(0.75) - df[selected_variable].quantile(0.25))
            outlier_count = ((df[selected_variable] < lower_bound) | (df[selected_variable] > upper_bound)).sum()
            st.write(f"**Number of Outliers for {selected_variable}:** {outlier_count}")

            # Boxplot for outliers
            st.write("**Boxplot of the Variable**")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_variable], ax=ax, color="lightcoral")
            ax.set_title(f'Boxplot of {selected_variable}', fontsize=14, weight='bold')
            st.pyplot(fig)

            # Missing values
            missing_values = df[selected_variable].isnull().sum()
            st.write(f"**Missing Values in {selected_variable}:** {missing_values}")

            # Additional statistics
            st.write("**Additional Statistics**")
            median = df[selected_variable].median()
            std_dev = df[selected_variable].std()
            skewness = df[selected_variable].skew()
            kurtosis = df[selected_variable].kurtosis()
            st.write(f"**Median:** {median}, **Standard Deviation:** {std_dev}, **Skewness:** {skewness}, **Kurtosis:** {kurtosis}")

            # Summary paragraph
            summary_paragraph = (
                f"The variable **{selected_variable}** exhibits a median value of approximately **{median:.2f}**, "
                f"indicating a central tendency in its distribution. With a standard deviation of around **{std_dev:.2f}**, "
                f"this suggests moderate variability around the median. The skewness value of **{skewness:.2f}** indicates "
                f"a slight positive asymmetry in the distribution, meaning there may be a longer tail on the right side, "
                f"implying that higher values are more prevalent. Additionally, the kurtosis of **{kurtosis:.2f}** suggests "
                f"that the distribution is slightly flatter than a normal distribution, indicating fewer extreme values. "
                f"Notably, there are **{missing_values}** missing entries in the dataset for this variable, which should be "
                f"addressed to enhance the robustness of any analysis or predictive modeling."
            )
            st.write(summary_paragraph)

            # Correlation with other features (only numeric columns)
            st.write("**Correlation with Other Features**")
            numeric_df = df.select_dtypes(include=['number'])  # Keep only numeric columns
            if not numeric_df.empty:
                correlations = numeric_df.corr()[selected_variable].sort_values(ascending=False)
                st.write(correlations)
            else:
                st.write("No numeric features available for correlation.")

        # Main EDA Section
        st.header('Exploratory Data Analysis')
        st.subheader('General information about the data')
        function_agent()

        # Variable Study
        st.header('Variable Study')
        user_question_variable = st.selectbox("Select a variable for study", [''] + df.columns.tolist())
        if user_question_variable:
            function_question_variable(user_question_variable)

        # DataFrame Study
        st.header('DataFrame Study')
        user_question_dataframe = st.text_area("Ask a question about the DataFrame")

        if user_question_dataframe:
            # Generate response based on the user's question
            dataframe_info = generate_text(user_question_dataframe)
            
            # Store the response in session state to prevent repetitions
            if "dataframe_responses" not in st.session_state:
                st.session_state["dataframe_responses"] = []
            
            if dataframe_info not in st.session_state["dataframe_responses"]:
                st.write(dataframe_info)
                st.session_state["dataframe_responses"].append(dataframe_info)

        # Predictive Modeling
        st.header('Predictive Modeling')
        st.write("**Select Feature Columns and Target Column for Prediction**")
        feature_cols = st.multiselect("Select Feature Columns", df.columns.tolist())
        target_col = st.selectbox("Select Target Column", df.columns.tolist())
        num_years = st.number_input("Enter number of years for prediction", min_value=1, value=1)
        model_choice = st.selectbox("Select Machine Learning Model", ["Linear Regression", "Random Forest", "Gradient Boosting", "Support Vector Regression"])

        if st.button("Train Model"):
            # Check if the target column and feature columns exist
            if target_col not in df.columns:
                st.error(f"Target column '{target_col}' does not exist in the dataset.")
            if not set(feature_cols).issubset(df.columns):
                st.error("One or more feature columns do not exist in the dataset.")
            else:
                # Handle missing values
                df = df.dropna(subset=[target_col] + feature_cols)

                X = df[feature_cols]
                y = df[target_col]

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Initialize the selected model
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Random Forest":
                    model = RandomForestRegressor()
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingRegressor()
                elif model_choice == "Support Vector Regression":
                    model = SVR()

                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                predictions = model.predict(X_test)

                # Calculate accuracy
                mse = mean_squared_error(y_test, predictions)
                rmse = mse ** 0.5
                st.write(f"**Model Accuracy (RMSE):** {rmse:.2f}")

                # Future prediction logic (you can customize this)
                future_prediction = model.predict(pd.DataFrame(X_test.iloc[-num_years:]))
                trend = "high" if future_prediction.mean() > y.mean() else "low"
                st.write(f"The market is expected to be {trend} in the future after {num_years} years.")

                # Display predictions
                st.write("**Predictions:**")
                st.write(predictions)