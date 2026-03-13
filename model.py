import cv2
import numpy as np
import tensorflow as tf
import os

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plant_recognition.keras')
class_names = ['Apple', 'Blueberry', 'Cherry', 'Grape', 'Pepper', 'Raspberry', 'Soybean', 'Strawberry', 'Tomato']


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    return tf.keras.models.load_model(model_path)

def predict(img_array):
    model = load_model(model_path)
    prediction = model.predict(img_array)
    return prediction


def predict_plant_health(model_name, plant_name, img_path):
    "przwiduje stan zdrowia rośliny i zwraca tylko nazwę klasy z naajlepszym dopasowaniem"

    health_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'health_models')
    health_class_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'health_classnames')
    model_file = os.path.join(health_model_path, model_name)
    health_classnames_file = os.path.join(health_class_folder_path, f"{plant_name}_health_classes.txt")

    if not os.path.exists(health_classnames_file):
        raise FileNotFoundError(f"Classnames file does not exist: {health_classnames_file}")

    with open(health_classnames_file, mode='r', encoding='utf-8') as file:
        health_class_names = {i: line.strip() for i, line in enumerate(file)}

    health_model = tf.keras.models.load_model(model_file) # załadowanie modelu

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File does not exist: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img_array = cv2.resize(img, (256, 256))
    img_array = np.expand_dims(img_array, axis=0)

    # predykcja
    health_prediction = health_model.predict(img_array)
    health_class_confidences = health_prediction[0]  # Lista z prawdopodobieństwami

    # Znalezienie indeksu klasy z najwyższym prawdopodobieństwem
    max_index = max(range(len(health_class_confidences)), key=lambda i: health_class_confidences[i])
    predicted_health_classname = health_class_names[max_index]
    return predicted_health_classname
