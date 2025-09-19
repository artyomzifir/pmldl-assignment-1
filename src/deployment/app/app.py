# code/deployment/app/app.py
import io
import os
import requests
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Demo", page_icon="🔢", layout="centered")

API_URL = os.getenv("API_URL", "http://api:8000")

st.title("MNIST 🔢 — нарисуй цифру")
st.caption("Белым по чёрному. Мы уменьшим до 28×28 и отправим на API.")

with st.sidebar:
    st.markdown("### Настройки")
    stroke = st.slider("Толщина кисти", 10, 60, 30, 2)
    bg_color = "#000000"
    stroke_color = "#FFFFFF"
    st.text(f"API: {API_URL}")

# Холст побольше для удобства; потом сожмём
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke,
    stroke_color=stroke_color,
    background_color=bg_color,
    width=280, height=280,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Очистить", type="secondary"):
        st.experimental_rerun()

def preprocess_to_28x28(rgba: np.ndarray) -> Image.Image:
    """
    rgba: [H,W,4] uint8. Переводим в grayscale, инвертировать не нужно,
    белое (кисть) остаётся ярким. Потом уменьшаем до 28×28.
    """
    # rgba → gray
    rgb = rgba[:, :, :3]
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    # Масштабируем до 28×28 (NEAREST, чтобы сохранить «пиксельность»)
    img = Image.fromarray(gray)
    img28 = img.resize((28, 28), Image.NEAREST)
    # Бинаризуем (по желанию): >0 → 255
    arr = np.array(img28)
    arr = (arr > 0).astype(np.uint8) * 255
    return Image.fromarray(arr)

pred_out = None
if canvas_result.image_data is not None:
    rgba = (canvas_result.image_data).astype("uint8")
    img28 = preprocess_to_28x28(rgba)

    # Показ превью (увеличим для удобства чтения)
    st.subheader("Предпросмотр 28×28")
    st.image(img28.resize((224, 224), Image.NEAREST), clamp=True)

    if st.button("Предсказать", type="primary"):
        # упакуем как PNG и отправим на API /predict (multipart)
        buf = io.BytesIO()
        img28.save(buf, format="PNG")
        buf.seek(0)
        files = {"file": ("digit.png", buf, "image/png")}
        try:
            r = requests.post(f"{API_URL}/predict", files=files, timeout=10)
            r.raise_for_status()
            pred_out = r.json()
        except Exception as e:
            st.error(f"Ошибка запроса: {e}")

if pred_out:
    st.subheader("Результат")
    st.write(f"**Класс:** {pred_out.get('pred')}")
    probs = pred_out.get("probs", [])
    if isinstance(probs, list) and len(probs) == 10:
        st.bar_chart({"probability": probs})
