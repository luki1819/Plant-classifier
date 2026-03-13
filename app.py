import os
import streamlit as st
import cv2
from PIL import Image
from video_processing import process_video
from photo_processing import predict_image
from figure import display_class_confidences
from model import predict_plant_health
import plotly.graph_objects as go

health_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'health_models')

st.set_page_config(page_title="Plant Classifier", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f7f7f7;
        }
        .title {
            text-align: center;
            color: #4CAF50;
            font-size: 3em;
            margin-bottom: 20px;
        }
        .spinner {
            color: #4CAF50;
        }
        .stImage {
            border-radius: 10px;
        }
        .prediction-info {
            font-size: 1.2em;
            color: #333333;
        }
        .prediction-header {
            font-size: 1.5em;
            font-weight: bold;
            color: #4CAF50;
        }
        .panel {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🌿Plant Classifier🌿</h1>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload Images or Videos of your plants", type=["jpg", "jpeg", "png", "mp4", "avi"],
                                  accept_multiple_files=True)

predictions = []

if uploaded_files:
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Temp')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_names = ["Select a file"] + [file.name for file in uploaded_files]
    selected_file_name = st.selectbox("Select a file to process", file_names)

    if selected_file_name != "Select a file":
        selected_file = next(file for file in uploaded_files if file.name == selected_file_name)
        file_ext = os.path.splitext(selected_file.name)[1].lower()

        if file_ext in [".jpg", ".jpeg", ".png"]:
            with st.spinner('Processing the image...'):
                image = Image.open(selected_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                img_path = os.path.join(save_dir, selected_file.name)
                image.save(img_path)

                predictions = predict_image(img_path)

                best_class, best_confidence = max(predictions, key=lambda x: x[1])
                st.markdown(f"<div class='prediction-header'>Prediction: {best_class}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='prediction-info'>Prediction accuracy: {round(float(best_confidence) * 100, 3)}%</div>", unsafe_allow_html=True)

                if predictions:
                    display_class_confidences(predictions)

                predicted_model_file = f"{best_class}_health.keras"
                full_model_path = os.path.join(health_model_path, predicted_model_file)

                if os.path.exists(full_model_path):
                    st.success(f"Model {predicted_model_file} found in 'health_models'.")
                    plant_health = predict_plant_health(predicted_model_file, best_class, img_path)
                    st.markdown(f"<div class='prediction-header'>Predicted health: {plant_health}</div>",
                                unsafe_allow_html=True)

                else:
                    st.warning(f"Model {predicted_model_file} not found in 'health_models'. \n It will be added in the future")

                os.remove(img_path)

        elif file_ext in [".mp4", ".avi"]:
            video_path = os.path.join(save_dir, selected_file.name)

            with open(video_path, "wb") as f:
                f.write(selected_file.read())

            st.write("It might take a while...")

            with st.spinner('Processing the video...'):
                best_frame_info = process_video(video_path, interval=2)

                frame = best_frame_info[1]
                best_class = best_frame_info[2]
                predictions = best_frame_info[6]
                best_confidence = best_frame_info[3]

                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            st.markdown(f"<div class='prediction-header'>Final Prediction:{best_class}", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction-info'>Prediction accuracy: {round(float(best_confidence) * 100, 3)}%</div>", unsafe_allow_html=True)

            if predictions:
                display_class_confidences(predictions)

            predicted_model_file = f"{best_class}_health.keras"
            full_model_path = os.path.join(health_model_path, predicted_model_file)

            if os.path.exists(full_model_path):
                st.success(f"Model {predicted_model_file} found in 'modele_health'.")
                plant_health = predict_plant_health(predicted_model_file, best_class, frame)
                st.markdown(f"<div class='prediction-header'>Predicted health: {plant_health}</div>",
                            unsafe_allow_html=True)

            else:
                st.warning(f"Model {predicted_model_file} not found in 'modele_health'.")

            os.remove(video_path)

        else:
            st.error("Unsupported file format!")




