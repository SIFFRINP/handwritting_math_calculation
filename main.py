import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from scripts.preprocessing import preprocess_images, preprocess_dataset, preprocess_image_main
import matplotlib.pyplot as plt

# Chemin d'accès
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)

image_path = os.path.join(project_root, "handwritting_math_calculation", "images", "NON.png")
print(image_path)

model_path = os.path.join(project_root, "handwritting_math_calculation", "models", "handwritten_math_calculator_model.keras")
print(model_path)

class_names = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=']

# Chargement du modèle
model = tf.keras.models.load_model(model_path)
print(f"Modèle chargé depuis : {model_path}")

# Chargement de l'image
image = load_img(image_path)
plt.imshow(image)
plt.axis('off')
plt.title("image avant pré-traitement")
plt.show()


image_clean = preprocess_image_main(image, 45, 45)


# _ TEST __________________________________________
# plt.imshow(image_clean.numpy().squeeze(), cmap='gray')
# plt.title("image après pré-traitement")
# plt.axis('off')
# plt.show()

# print("Forme de l'image avant prédiction :", image_clean.shape)

# Prédiction de l'image
predictions = model.predict(image_clean)
predicted_class_index = np.argmax(predictions, axis=1)[0]
confidence = np.max(predictions) * 100
predicted_class = class_names[predicted_class_index]

print(f"Le symbole prédit est : {predicted_class} avec une confiance de {confidence:.2f}%")
