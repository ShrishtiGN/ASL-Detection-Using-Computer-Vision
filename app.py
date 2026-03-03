import streamlit as st
import cv2
import numpy as np
import time
from simple_asl_detector import SimpleASLDetector
from image_processor import ImageProcessor

# Page configuration
st.set_page_config(
    page_title="ASL Sign Language Detector",
    page_icon="🤟",
    layout="wide"
)

# Load CSS
with open('static/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_local_html=True)

def main():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🤟 ASL Sign Language Detector")
    st.markdown("Real-time American Sign Language recognition using Computer Vision")
    st.markdown('</div>', unsafe_allow_html=True)

    # Initialize detector and processor
    if 'detector' not in st.session_state:
        st.session_state.detector = SimpleASLDetector()
    if 'processor' not in st.session_state:
        st.session_state.processor = ImageProcessor()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="detection-card">', unsafe_allow_html=True)
        placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="detection-card">', unsafe_allow_html=True)
        st.subheader("Detection Results")
        result_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    # Simple loop for processing (Streamlit runs this)
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame, prediction, confidence = st.session_state.detector.process_frame(frame)
        
        # Display results
        placeholder.image(processed_frame, channels="BGR", use_container_width=True)
        
        result_placeholder.markdown(f"""
            <div class="prediction-item">
                <h3>Detected: {prediction}</h3>
                <p>Confidence: {confidence:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        
        time.sleep(0.01)

    cap.release()

if __name__ == "__main__":
    main()