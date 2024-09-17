import pandas as pd
import joblib
import streamlit as st
import os
import gdown  # Import gdown to download from Google Drive
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import base64

# Google Drive links to the models
CROP_YIELD_MODEL_DRIVE_LINK = "https://drive.google.com/uc?id=13SRAcz2IASy_kHzju0Xq4cr7nuek9DJ0"

MODEL_PATH = 'crop_yield_model.pkl'


# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive...")
        gdown.download(CROP_YIELD_MODEL_DRIVE_LINK, MODEL_PATH, quiet=False)
        st.write("Download complete")

# Call the function to download the model
download_model()

# The rest of your existing code follows...


# Function to load and encode the image as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    return encoded

# Function to add background image from local path
def set_background(image_path):
    image_base64 = get_base64_image(image_path)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to display logo and title in a row
def display_logo_and_title(logo_path, title_text):
    logo_base64 = get_base64_image(logo_path)
    st.markdown(
        f'''
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
            <img src="data:image/jpeg;base64,{logo_base64}" alt="Logo">
            <h1>{title_text}</h1>
        </div>
        ''', unsafe_allow_html=True
    )

# Function to add a footer with specific styling
def add_footer():
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
            padding: 5px 0;
            font-size: 14px;
        }
        </style>
        <div class="footer">
            @2024 - Agro Techies
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to train and save the model
def train_and_save_model():
    st.write("Loading dataset...")
    # Load the dataset
    data = pd.read_csv('crop_yield_cleaned.csv')

    st.write("Cleaning data...")
    # Clean the categorical columns by removing extra spaces
    data['Crop'] = data['Crop'].str.strip()
    data['Season'] = data['Season'].str.strip()
    data['State'] = data['State'].str.strip()

    # Check for missing values and drop them
    data.dropna(inplace=True)

    # Define features (X) and target (y)
    X = data.drop(columns=['Yield'])  # 'Yield' is the target column
    y = data['Yield']

    st.write("Setting up preprocessing pipeline...")
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Crop', 'Season', 'State'])
        ])

    st.write("Setting up the full pipeline...")
    # Create the full pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))  # Parallelize
    ])

    st.write("Splitting data into training and testing sets...")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("Training the model...")
    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    st.write("Evaluating the model on the test set...")
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error on Test Set: {mse:.2f}")
    st.write(f"R² Score on Test Set: {r2:.2f}")

    # If the R² score is low, there might be issues with the model
    if r2 < 0.1:
        st.warning("Model may not be predicting well. Consider checking the input data or model parameters.")

    st.write("Saving the trained model...")
    # Save the trained model
    joblib.dump(pipeline, 'crop_yield_model.pkl')
    st.success("Model trained and saved as 'crop_yield_model.pkl'")

# Function to load the model
@st.cache_resource  # Cache the model loading
def load_model():
    return joblib.load('crop_yield_model.pkl')

# Streamlit application
def main():
    # Set background and display logo and title
    set_background("background.jpg")
    display_logo_and_title('logo.jpg', 'Agro Techies Yield Predictor')

    # Check if the model file exists; if not, train the model
    if not os.path.isfile('crop_yield_model.pkl'):
        st.warning("Model file not found. Training a new model. This may take a while...")
        train_and_save_model()

    # Load the saved model
    pipeline = load_model()

    # Dropdown options based on the unique values from crop_yield.csv
    crops = ['Arecanut', 'Arhar/Tur', 'Castor seed', 'Coconut', 'Cotton(lint)', 'Dry chillies', 'Gram', 'Jute', 
             'Linseed', 'Maize', 'Mesta', 'Niger seed', 'Onion', 'Other Rabi pulses', 'Potato', 
             'Rapeseed & Mustard', 'Rice', 'Sesamum', 'Small millets', 'Sugarcane', 'Sweet potato', 
             'Tapioca', 'Tobacco', 'Turmeric', 'Wheat', 'Bajra', 'Black pepper', 'Cardamom', 
             'Coriander', 'Garlic', 'Ginger', 'Groundnut', 'Horse-gram', 'Jowar', 'Ragi', 'Cashewnut', 
             'Banana', 'Soyabean', 'Barley', 'Khesari', 'Masoor', 'Moong(Green Gram)', 'Other Kharif pulses', 
             'Safflower', 'Sannhamp', 'Sunflower', 'Urad', 'Peas & beans (Pulses)', 'Other oilseeds', 
             'Other Cereals', 'Cowpea(Lobia)', 'Oilseeds total', 'Guar seed', 'Other Summer Pulses', 'Moth']
    
    seasons = ['Whole Year', 'Kharif', 'Rabi', 'Autumn', 'Summer', 'Winter']
    
    states = ['Assam', 'Karnataka', 'Kerala', 'Meghalaya', 'West Bengal', 'Puducherry', 'Goa', 
              'Andhra Pradesh', 'Tamil Nadu', 'Odisha', 'Bihar', 'Gujarat', 'Madhya Pradesh', 
              'Maharashtra', 'Mizoram', 'Punjab', 'Uttar Pradesh', 'Haryana', 'Himachal Pradesh', 
              'Tripura', 'Nagaland', 'Chhattisgarh', 'Uttarakhand', 'Jharkhand', 'Delhi', 'Manipur', 
              'Jammu and Kashmir', 'Telangana', 'Arunachal Pradesh', 'Sikkim']

    # Center the input form without "Input Features" text
    with st.form(key="input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            crop = st.selectbox("Select Crop:", crops)
            season = st.selectbox("Select Season:", seasons)
            state = st.selectbox("Select State:", states)
        
        with col2:
            area = st.number_input("Area (in hectares):", min_value=0.0, format="%.2f")
            production = st.number_input("Production (in tons):", min_value=0.0, format="%.2f")
            rainfall = st.number_input("Annual Rainfall (in mm):", min_value=0.0, format="%.2f")
            fertilizer = st.number_input("Fertilizer (in kg/ha):", min_value=0.0, format="%.2f")
            pesticide = st.number_input("Pesticide (in kg/ha):", min_value=0.0, format="%.2f")
        
        submit_button = st.form_submit_button("Predict Yield")
        
        if submit_button:
            with st.spinner("Predicting..."):
                # Strip any extra spaces from user input
                crop = crop.strip()
                season = season.strip()
                state = state.strip()

                # Create a DataFrame from the input values
                input_data = pd.DataFrame({
                    'Crop': [crop],
                    'Season': [season],
                    'State': [state],
                    'Area': [area],
                    'Production': [production],
                    'Annual_Rainfall': [rainfall],
                    'Fertilizer': [fertilizer],
                    'Pesticide': [pesticide]
                })

                # Make prediction
                prediction = pipeline.predict(input_data)[0]
                st.success(f'Predicted Yield: {prediction:.2f} tons per hectare')

    # Add footer
    add_footer()

if __name__ == "__main__":
    main()
