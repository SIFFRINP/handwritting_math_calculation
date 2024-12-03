print("Hello, World!")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

import tensorflow as tf
tf.debugging.set_log_device_placement(True)



def preprocess_image(image_path):
    # Charger et redimensionner l'image
    img = load_img(image_path, color_mode='rgb', target_size=(45, 45))
    img_array = img_to_array(img)

    # Conversion en niveaux de gris et normalisation
    img_array = tf.image.rgb_to_grayscale(img_array)  # Conversion en niveaux de gris
    img_array = tf.cast(img_array, tf.float32) / 255.0  # Normalisation entre 0 et 1

    # Ajouter une dimension pour simuler un batch
    img_array = np.expand_dims(img_array, axis=0)

    # # Affichage pour vérifier l'image prétraitée
    # plt.imshow(img_array.squeeze(), cmap='gray')
    # plt.title("Image prétraitée")
    # plt.axis('off')
    # plt.show()

    return img_array

def predict_digit(model_path, image_path, class_names):
    # Charger le modèle
    model = load_model(model_path)
    print(f"Modèle chargé depuis : {model_path}")

    # Préparer l'image
    image = preprocess_image(image_path)

    print(f"Forme : {image.shape}, Valeurs min/max : {image.min()}/{image.max()}")


    # Faire une prédiction
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions, axis=-1)[0]
    confidence = np.max(predictions) * 100
    predicted_class = class_names[predicted_label]

    print(f"Le symbole prédit est : {predicted_class} avec une confiance de {confidence:.2f}%")


# Classes à prédire (remplace-les par les vraies classes de ton dataset)
class_names = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'C',
'Delta', 'G', 'H', 'M', 'N', 'R', 'S', 'T', 'X', '[', ']', 'alpha', 'ascii_124', 'b', 'beta',
'cos', 'd', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'gamma', 'geq', 'gt', 'i',
'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log', 'lt', 'mu', 'neq',
'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'rightarrow', 'sigma', 'sin', 'sqrt', 'sum', 'tan',
'theta', 'times', 'u', 'v', 'w', 'y', 'z', '{', '}']

# Tester avec une image
predict_digit("models/handwritten_math_calculator_model.keras", "images/OUI.png", class_names)


