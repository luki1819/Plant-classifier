import os.path
import cv2
import numpy as np
from model import predict, class_names


def load_image(img_path, img_height=256, img_width=256):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File does not exist: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img_resized = cv2.resize(img, (img_width, img_height))
    return img_resized


def predict_image(img_path):
    img_resized = load_image(img_path)
    img_array = np.expand_dims(img_resized, axis=0)
    prediction = predict(img_array)
    class_confidences = prediction[0]
    predicted_classes = [class_names[i] for i in range(len(class_confidences))]
    # tworzenie listy krotek z klasą i jej dopasowaniem
    class_predictions = list(zip(predicted_classes, class_confidences))
    return class_predictions
