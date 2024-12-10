import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from configuration import BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Ajoute un flip vertical
#     tf.keras.layers.RandomRotation(0.4),  # Augmente l'amplitude de rotation
#     tf.keras.layers.RandomZoom(height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3)),  # Zoom plus large
#     tf.keras.layers.RandomTranslation(0.2, 0.2),  # Décalages horizontaux et verticaux
#     tf.keras.layers.RandomContrast(0.2)  # Ajoute des variations de contraste
# ])


def preprocess_images(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def preprocess_dataset(dataset, batch_size, augment=False):
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def preprocess_image_main(image, img_height, img_width):
    # Si c'est un tableau NumPy
    if isinstance(image, np.ndarray): 
        # Si l'image a trois canaux (couleur), convertir en niveaux de gris
        if image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)

        
    # Sinon, on suppose que c'est une image PIL
    else:  
        image = image.resize((img_height, img_width))
        image = img_to_array(image) # Convertir en tableau NumPy
        image = tf.image.rgb_to_grayscale(image)

    # Normalisation et ajout de dimension batch
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image


