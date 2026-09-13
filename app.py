import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(page_title="MNIST Vision Dashboard", layout="wide", page_icon="👁️")

@st.cache_resource
def load_vision_model():
    return load_model("mnist_model.keras")

def preprocess_camera(image_array):
    """Pipeline for messy, real-world camera photos."""
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    # Blur and threshold to fix room lighting and camera grain
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50: 
        return None  

    x, y, w, h = cv2.boundingRect(c)
    digit = thresh[y:y + h, x:x + w]

    # Normalize to 20x20 bounding box
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    digit = cv2.dilate(digit, np.ones((2, 2), np.uint8), iterations=1)

    # Center on 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = digit

    return canvas

def preprocess_canvas(canvas_image):
    """Pipeline for pure digital ink."""
    # Extract the Red channel (0 for black background, 255 for white ink)
    img = canvas_image[:, :, 0].astype(np.uint8)
    
    # Skip blur and thresholding entirely. Go straight to contour detection.
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 10: 
        return None  
        
    x, y, w, h = cv2.boundingRect(c)
    digit = img[y:y + h, x:x + w]
    
    # Normalize to 20x20 bounding box
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center on 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = digit
    
    return canvas

# --- UI Layout ---
st.title("👁️ MNIST Real-Time Vision Pipeline")
st.markdown("Test the neural network using your camera or the digital drawing canvas.")

model = load_vision_model()

with st.sidebar:
    st.header("Input Method")
    input_mode = st.radio("Choose how to input the digit:", ("Digital Canvas", "Webcam Capture"))

col1, col2, col3 = st.columns([1.5, 1, 1.5])
processed_img = None

with col1:
    st.subheader("1. Data Input")
    
    if input_mode == "Digital Canvas":
        st.write("Draw a single digit (0-9) below:")
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=20, # Increased thickness for better downscaling
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=300,
            height=300,
            drawing_mode="freedraw",
            key="canvas",
            return_image_data=True,
        )
        # Process directly if the canvas is not blank
        if canvas_result.image_data is not None and np.sum(canvas_result.image_data[:, :, 0]) > 0:
            processed_img = preprocess_canvas(canvas_result.image_data)

    else:
        st.write("Hold a piece of paper with a digit exactly in the **center** of the camera.")
        camera_photo = st.camera_input("Capture Digit")
        if camera_photo is not None:
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            
            # --- THE FIX: Center Region of Interest (ROI) Crop ---
            # Crop a central square to ignore the background room data
            h, w = raw_img.shape[:2]
            box_size = int(h * 0.5)  # Use the middle 50% of the image height
            y1 = (h - box_size) // 2
            x1 = (w - box_size) // 2
            
            roi_img = raw_img[y1:y1 + box_size, x1:x1 + box_size]
            
            # Show the user exactly what got cropped so they can adjust the paper
            st.image(roi_img, caption="Cropped Camera View (ROI)", width=150)
            
            processed_img = preprocess_camera(roi_img)

with col2:
    st.subheader("2. Model Input")
    if processed_img is not None:
        st.image(processed_img, caption="28x28 Tensor (Enlarged)", width=150, clamp=True)
        st.caption("This normalized array is exactly what the neural network sees.")
    else:
        st.info("Awaiting clear input...")

with col3:
    st.subheader("3. Neural Network Output")
    if processed_img is not None:
        input_data = processed_img.astype("float32") / 255.0
        input_data = np.expand_dims(input_data, axis=(0, -1))

        prediction = model.predict(input_data, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.metric(label="Predicted Digit", value=str(digit))
        
        st.write("Confidence Score:")
        st.progress(float(confidence))
        st.caption(f"{confidence * 100:.2f}% certainty")