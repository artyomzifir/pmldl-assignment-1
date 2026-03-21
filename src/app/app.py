"""Streamlit frontend for drawing a digit and sending it to the API.

The UI logic is wrapped in functions so the module can be imported safely by
Sphinx during documentation builds.
"""

import numpy as np
import requests
from PIL import Image

API_URL = "http://api:8085"


def preprocess_to_28x28(rgba: np.ndarray) -> np.ndarray:
    """Convert the Streamlit canvas image into a binary ``28 x 28`` array.

    Args:
        rgba: Canvas image represented as an RGBA numpy array.

    Returns:
        np.ndarray: A binary image with shape ``28 x 28``.
    """
    rgb = rgba[:, :, :3]
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    img = Image.fromarray(gray).resize((28, 28), Image.NEAREST)
    arr = np.array(img)

    # The API expects a simple binary image rather than grayscale intensities.
    return (arr > 0).astype(np.uint8)


def request_prediction(arr28: np.ndarray, api_url: str = API_URL) -> dict[str, object]:
    """Send the preprocessed digit image to the FastAPI service.

    Args:
        arr28: Binary digit image with shape ``28 x 28``.
        api_url: Base URL of the prediction service.

    Returns:
        dict[str, object]: Parsed JSON response from the API.
    """
    response = requests.post(
        f"{api_url}/predict",
        json={"data": arr28.tolist()},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def render_app(api_url: str = API_URL) -> None:
    """Render the interactive Streamlit application.

    Args:
        api_url: Base URL of the FastAPI backend.
    """
    import streamlit as st
    from streamlit_drawable_canvas import st_canvas

    st.set_page_config(page_title="MNIST CNN Predictor", layout="centered")
    st.title("MNIST CNN Predictor")
    st.write("Draw a digit from 0 to 9 and press Predict.")

    with st.sidebar:
        stroke = st.slider("Brush thickness", 10, 60, 30, 2)
        bg_color = "#000000"
        stroke_color = "#FFFFFF"
        st.text(f"API: {api_url}")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke,
        stroke_color=stroke_color,
        background_color=bg_color,
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    pred_out = None

    if canvas_result.image_data is not None:
        rgba = canvas_result.image_data.astype("uint8")
        arr28 = preprocess_to_28x28(rgba)

        st.subheader("Preview 28 x 28")
        preview = Image.fromarray(arr28 * 255).resize((224, 224), Image.NEAREST)
        st.image(preview, clamp=True)

        if st.button("Predict", type="primary"):
            try:
                pred_out = request_prediction(arr28, api_url=api_url)
            except Exception as exc:
                st.error(f"Error while calling the API: {exc}")

    if pred_out is not None:
        st.subheader("Result")
        st.write(f"**Digit:** {pred_out.get('prediction')}")
        st.bar_chart({"Probability": pred_out["probs"]})


def main() -> None:
    """Entry point used by ``streamlit run``."""
    render_app()


if __name__ == "__main__":
    main()
