import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array


def preprocess_images(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def preprocess_dataset(dataset, batch_size=32):
    dataset = dataset.map(preprocess_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def preprocess_image_main(image, img_height=45, img_width=45):
    image = image.resize((img_height, img_width)) 
    image = img_to_array(image)  # Convertir en tableau NumPy
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32) / 255.0  # Normaliser
    image = tf.expand_dims(image, axis=0)
    return image

