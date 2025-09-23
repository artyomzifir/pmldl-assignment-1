import requests
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np

st.set_page_config(page_title="MNIST CNN Predictor", layout="centered")

API_URL = "http://api:8085"

st.title("MNIST CNN Predictor")
st.write("Draw a digit (0-9) below and click 'Predict' to see the model's prediction.")

with st.sidebar:
    stroke = st.slider("Brush thickness", 10, 60, 30, 2)
    bg_color = "#000000"
    stroke_color = "#FFFFFF"
    st.text(f"API: {API_URL}")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke,
    stroke_color=stroke_color,
    background_color=bg_color,
    width=280, height=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_to_28x28(rgba: np.ndarray) -> np.ndarray:
    """Преобразуем canvas в картинку 28x28"""
    rgb = rgba[:, :, :3]
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    img = Image.fromarray(gray).resize((28, 28), Image.NEAREST)
    arr = np.array(img)
    arr = (arr > 0).astype(np.uint8)
    return arr

pred_out = None
if canvas_result.image_data is not None:
    rgba = (canvas_result.image_data).astype("uint8")
    arr28 = preprocess_to_28x28(rgba)

    st.subheader("Preview (28x28)")
    st.image(Image.fromarray(arr28*255).resize((224, 224), Image.NEAREST), clamp=True)

    if st.button("Predict", type="primary"):
        try:
            r = requests.post(f"{API_URL}/predict", json={"data": arr28.tolist()}, timeout=10)
            r.raise_for_status()
            pred_out = r.json()
        except Exception as e:
            st.error(f"Error with api or model: {e}")

if pred_out is not None:
    st.subheader("Result")
    st.write(f"**Digit:** {pred_out.get('prediction')}")
    st.bar_chart({"Probability": pred_out["probs"]})
