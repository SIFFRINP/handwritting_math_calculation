import os
import tensorflow as tf
import numpy as np


def load_data(data_dir, img_height=45, img_width=45, batch_size=None):
    # Charger les données
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(img_height, img_width),
        labels='inferred',
        batch_size=batch_size
    )
    
    class_names = dataset.class_names
    print(f"Classes trouvées : {class_names}")

    for image, label in dataset.take(1):
        print(f"Première image shape : {image.shape}, Étiquette : {label}")
    
    return dataset, class_names