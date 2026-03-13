import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.python.ops.confusion_matrix import confusion_matrix
import seaborn as sns

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train')
test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')

def plot_history(history):
    # do strat
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # do dokładności
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.show()


batch_size = 32
# wybór rozmiarów zdjęcia
img_height = 256
img_width = 256

# Tworzenie zestawów danych
validation_split = 0.2

# Wczytanie danych treningowych
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=validation_split,
    subset="training",
    seed=123
)

# Wczytanie danych walidacyjnych
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=validation_split,
    subset="validation",
    seed=123
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="int",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Sprawdzenie klasy i etykiety
class_names = train_dataset.class_names
print("Nazwy klas:", class_names)


def convert_to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)  # Konwersja do grayscale (kształt: [height, width, 1])
    return image, label


# Mapowanie konwersji na grayscale
train_dataset = train_dataset.map(convert_to_grayscale)
validation_dataset = validation_dataset.map(convert_to_grayscale)
test_dataset = test_dataset.map(convert_to_grayscale)

# Sprawdzenie kształtu obrazu i etykiet przykładowego obrazu
for images, labels in train_dataset.take(1):
    print("Kształt obrazów po konwersji:", images.shape)
    print("Etykiety:", labels)


model_path = 'plant_recognition.keras'
if os.path.exists(model_path):
    print("Model found. Loading the saved model...")
    model = tf.keras.models.load_model(model_path)
else:
    print("No saved model found. Training a new model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_dataset, epochs=20, validation_data=validation_dataset)

    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    model.save(model_path)
    print(f"Model trained and saved as {model_path}")

    plot_history(history)

    #predykcja na zbiorze testowym:
    true_labels = []
    predictions = []

    # Przechodzenie przez zbiór testowy
    for images, labels in test_dataset:
        # Uzyskanie przewidywań dla bieżącej partii
        preds = model.predict(images)

        # Dodanie etykiet do list
        true_labels.extend(labels.numpy())  # Prawdziwe etykiety
        predictions.extend(np.argmax(preds, axis=1))  # Przewidywania modelu (indeksy klas)

    # Konwersja do numpy
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    cm = confusion_matrix(true_labels, predictions)

    # Wyświetlanie macierzy pomyłek
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predykcje')
    plt.ylabel('Prawdziwe etykiety')
    plt.title('Macierz Pomyłek')
    plt.show()
