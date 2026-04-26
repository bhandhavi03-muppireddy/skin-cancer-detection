import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Smart Derma Trace", layout="centered")

@st.cache_resource
def get_model():
    return load_model("skin_cancer_model.h5", compile=False)

model = get_model()

def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)

def preprocess(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (128, 128))
    image = remove_hair(image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def go(page):
    st.session_state.page = page
    st.rerun()

# WELCOME PAGE
if st.session_state.page == "welcome":
    st.title("Welcome to Smart Derma Trace")
    st.subheader("AI Powered Skin Cancer Detection System")
    st.write("This application helps users upload a skin image and check whether skin cancer is detected or not.")
    
    if st.button("Continue"):
        go("login")

# LOGIN PAGE
elif st.session_state.page == "login":
    st.title("Login")

    username = st.text_input("Username / Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            go("home")
        else:
            st.error("Please enter username and password")

# HOME PAGE
elif st.session_state.page == "home":
    st.title("Smart Derma Trace")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Skin Cancer Detection"):
            go("detection")

    with col2:
        if st.button("Awareness Page"):
            go("awareness")

# DETECTION PAGE
elif st.session_state.page == "detection":
    st.title("Skin Cancer Detection")

    if st.button("Back"):
        go("home")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        x = preprocess(img)
        pred = float(model.predict(x, verbose=0)[0][0])

        st.write(f"Prediction score: {pred:.4f}")

        if pred > 0.5:
            st.error("Skin Cancer Detected")
            st.warning("Please consult a dermatologist immediately.")
        else:
            st.success("Skin Cancer Not Detected")
            st.info("The uploaded image is likely benign.")

# AWARENESS PAGE
elif st.session_state.page == "awareness":
    st.title("Skin Health Awareness")

    if st.button("Back"):
        go("home")

    st.write("""
    - Skin cancer can be treated better when detected early.
    - Avoid direct sunlight for long hours.
    - Use sunscreen with SPF 30 or higher.
    - Wear protective clothing, sunglasses, and hats.
    - Check your skin regularly for new or changing spots.
    - Consult a dermatologist if a mole changes in size, color, or shape.
    - This app is only for awareness and screening, not final medical diagnosis.
    """)
