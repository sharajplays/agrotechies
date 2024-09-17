import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown  # Import gdown to download from Google Drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Google Drive links to the models
RANDOM_FOREST_MODEL_DRIVE_LINK = "https://drive.google.com/uc?id=12LoaNsy9xamYP4aOqytXqD-7pK-pPtFk"

MODEL_PATH = 'random_forest_model.joblib'

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive...")
        gdown.download(RANDOM_FOREST_MODEL_DRIVE_LINK, MODEL_PATH, quiet=False)
        st.write("Download complete")

# Call the function to download the model
download_model()

# The rest of your existing code follows...

# Function to add background image from local path
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background from specified path
add_bg_from_local('background.jpg')

# Function to add the logo and title
def add_logo_and_title(logo_path, title_text):
    with open(logo_path, "rb") as logo_file:
        encoded_logo = base64.b64encode(logo_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .logo-container {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
            margin-bottom: 20px;
        }}
        .logo-container img {{
            border-radius: 50%;
            width: 60px;
            height: 60px;
        }}
        .logo-container h1 {{
            font-size: 18px;
            margin-left: 10px;
            color: black;
        }}
        </style>
        <div class="logo-container">
            <img src="data:image/jpg;base64,{encoded_logo}" alt="Logo">
            <h1>{title_text}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Add logo and title at the top
add_logo_and_title('logo.jpg', 'Agro Techies Crops Price Predictor')

# Cache the data loading function
@st.cache_data
def load_data():
    data = pd.read_csv('dataset.csv') 
    data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
    data['month'] = data['date'].dt.month
    return data

# Load the data
data = load_data()

# Prepare the features and target
features = data[['state', 'crops', 'month']]
target = data['modal_price']

# Encode categorical variables
label_encoders = {}
for column in ['state', 'crops']:
    label_encoders[column] = LabelEncoder()
    features[column] = label_encoders[column].fit_transform(features[column])

# Scale the 'month' feature
scaler = StandardScaler()
features[['month']] = scaler.fit_transform(features[['month']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Load or train the model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.write("Model loaded from file")
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    st.write("Model trained and saved to file")

# Custom CSS for smaller footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 5px 0;  /* Reduced padding to make footer smaller */
        font-size: 14px;  /* Smaller footer font size */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User input fields for prediction
state = st.selectbox('Select State', data['state'].unique())
crop = st.selectbox('Select Crop', data['crops'].unique())
month = st.selectbox('Select month in which you want to predict', range(1, 13))

# Prediction button
if st.button('Predict'):
    # Encode the user inputs
    state_encoded = label_encoders['state'].transform([state])[0]
    crop_encoded = label_encoders['crops'].transform([crop])[0]
    month_scaled = scaler.transform([[month]])[0][0]

    # Create input array
    input_data = np.array([[state_encoded, crop_encoded, month_scaled]])

    # Predict the price
    predicted_price = model.predict(input_data)[0]

    # Display the predicted price
    st.markdown(
        f"""
        <div style='
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
            border: 1px solid black;
            background-color: white;
            color: black;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        '>
            Predicted Modal Price: ₹{predicted_price:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Evaluate the model and display metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model Mean Squared Error: {mse:.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    bins = np.linspace(min(target), max(target), 10)
    y_test_binned = np.digitize(y_test, bins)
    y_pred_binned = np.digitize(y_pred, bins)
    conf_matrix = confusion_matrix(y_test_binned, y_pred_binned)

    # Plot the confusion matrix with reduced size
    fig, ax = plt.subplots(figsize=(8, 6))  # Reduced size
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# Footer
st.markdown(
    """
    <div class="footer">
        @2024 - Agro Techies
    </div>
    """,
    unsafe_allow_html=True
)

