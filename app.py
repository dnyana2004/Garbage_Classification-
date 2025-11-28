import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np

st.set_page_config(page_title="Garbage Classification", layout="centered")

st.title("Garbage Classification")
st.write("Upload an image")

# Load model
model = tf.keras.models.load_model("best_model.h5")

# Update labels if needed
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    pred_class = labels[np.argmax(pred)]

    st.success(f"Predicted Class: **{pred_class}**")
