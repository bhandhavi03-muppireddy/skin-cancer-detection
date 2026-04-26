import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Skin Cancer Detector", layout="centered")

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

st.title("Skin Cancer Detection")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    x = preprocess(img)
    pred = float(model.predict(x, verbose=0)[0][0])

    st.write(f"Prediction score: {pred:.4f}")
    if pred > 0.5:
        st.error("Possible cancer")
    else:
        st.success("Likely benign")