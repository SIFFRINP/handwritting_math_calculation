print("Hello, World!")

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Affichage
    plt.imshow(img_array.squeeze(), cmap='gray')
    plt.title("Image prétraitée")
    plt.axis('off')
    plt.show()

    return img_array

def predict_digit(model_path, image_path):
    # Charger le modèle
    model = load_model(model_path)
    print(f"Modèle chargé depuis : {model_path}")

    # Préparer l'image
    image = preprocess_image(image_path)


    # Faire une prédiction
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions, axis=-1)[0]
    confidence = np.max(predictions) * 100


    print(f"Le chiffre prédit est : {predicted_label} avec une confiance de {confidence:.2f}%")

# Tester avec une image
predict_digit("models/mnist_model.h5", "images/chiffre3.png")
