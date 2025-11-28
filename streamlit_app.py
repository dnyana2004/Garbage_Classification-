import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🗑️ Garbage Classification App")

model = tf.keras.models.load_model("best_model.h5")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

class_names = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC", "TRASH"]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    st.write("### ♻️ Prediction:", class_names[class_idx])
