import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
from tensorflow.keras.preprocessing.image import img_to_array
from configuration import batch_size, img_height, img_width


def data_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image


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
    if isinstance(image, np.ndarray):  # Si c'est un tableau NumPy
        # Si l'image a trois canaux (couleur), convertir en niveaux de gris
        if image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)
    else:  # Sinon, on suppose que c'est une image PIL
        image = image.resize((img_height, img_width))
        image = img_to_array(image) # Convertir en tableau NumPy
        image = tf.image.rgb_to_grayscale(image)
    # Normalisation et ajout de dimension batch
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image



def downsample_classes(data_dir, target_size):
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if len(images) > target_size:
                # Sélectionner aléatoirement `target_size` images
                images_to_keep = random.sample(images, target_size)
                # Supprimer les autres images
                for img in images:
                    if img not in images_to_keep:
                        os.remove(os.path.join(class_path, img))
                print(f"Downsampled {class_name} to {target_size} images.")


def augment_classes(data_dir, target_size, augment_function):
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if len(images) < target_size:
                deficit = target_size - len(images)
                print(f"Augmenting {class_name} by {deficit} images.")
                for i in range(deficit):
                    img_name = random.choice(images)
                    img_path = os.path.join(class_path, img_name)
                    img = tf.keras.preprocessing.image.load_img(img_path)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    augmented_img = augment_function(img_array)
                    # Sauvegarder l'image augmentée
                    augmented_img = tf.keras.preprocessing.image.array_to_img(augmented_img)
                    augmented_img.save(os.path.join(class_path, f"aug_{i}.png"))



def print_dataset_distribution(data_dir):
    distribution = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            distribution[class_name] = num_images
    print("Répartition du dataset :")
    for class_name, count in distribution.items():
        print(f"Classe {class_name}: {count} images")
