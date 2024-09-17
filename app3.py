import streamlit as st
import subprocess
import base64

# Function to add background image using base64 encoding
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Add background
add_bg_from_local("background.jpg")  # Adjust the path to your background image

# Function to add the logo, title, and footer
def add_logo_title_footer(logo_path, title_text, footer_text):
    with open(logo_path, "rb") as logo_file:
        encoded_logo = base64.b64encode(logo_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .logo-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }}
        .logo-container img {{
            border-radius: 50%;
            width: 60px;
            height: 60px;
        }}
        .logo-container h1 {{
            font-size: 24px;
            margin-left: 10px;
            color: black;
        }}
        .footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: white;
            color: black;
            text-align: center;
            padding: 5px 0; /* Reduced padding for a smaller footer */
            font-size: 14px;
        }}
        </style>
        <div class="logo-container">
            <img src="data:image/png;base64,{encoded_logo}" alt="Logo">
            <h1>{title_text}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add footer
    st.markdown(
        f"""
        <div class="footer">
            {footer_text}
        </div>
        """,
        unsafe_allow_html=True
    )

# Add logo, title, and footer
add_logo_title_footer(
    'logo.jpg',  # Path to your logo
    'Agro Techies Services',  # Title text
    '@2024 - Agro Techies'  # Footer text
)

# Custom CSS for hover effect on images and rounded corners
st.markdown("""
    <style>
    .hover-effect {
        transition: transform .2s;
        border-radius: 15px; /* Rounded corners for the images */
        cursor: pointer;
    }
    .hover-effect:hover {
        transform: scale(1.05);
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
    }
    .app-image {
        width: 200px;  /* Keep image size */
        height: 200px; /* Keep image size */
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 10px;
        border-radius: 15px; /* Ensure rounded corners */
    }
    .center-text {
        text-align: center;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Create two side-by-side columns for the two apps
col1, col2 = st.columns(2)

# App 1 image (clickable)
with col1:
    app1_image_path = 'app1_image.png'
    app1_image = base64.b64encode(open(app1_image_path, 'rb').read()).decode()

    # Display App 1 image
    st.markdown(f"""
    <div class="center-text">
        <img src="data:image/png;base64,{app1_image}" class="hover-effect app-image" onclick="window.location.href='/'" />
        <p>App 1: Crop Price Predictor</p>
    </div>
    """, unsafe_allow_html=True)

# App 2 image (clickable)
with col2:
    app2_image_path = 'app2_image.png'
    app2_image = base64.b64encode(open(app2_image_path, 'rb').read()).decode()

    # Display App 2 image
    st.markdown(f"""
    <div class="center-text">
        <img src="data:image/png;base64,{app2_image}" class="hover-effect app-image" onclick="window.location.href='/'" />
        <p>App 2: Crop Yield Predictor</p>
    </div>
    """, unsafe_allow_html=True)

# Add a selectbox to simulate clicking the images
selected_app = st.selectbox(
    "Select an app to run",
    options=["None", "App 1: Crop Price Predictor", "App 2: Crop Yield Predictor"],
    index=0
)

# Check which app was selected and launch it
if selected_app == "App 1: Crop Price Predictor":
    subprocess.Popen(["streamlit", "run", "app1.py"])
elif selected_app == "App 2: Crop Yield Predictor":
    subprocess.Popen(["streamlit", "run", "app2.py"])
