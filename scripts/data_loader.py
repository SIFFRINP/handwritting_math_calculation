import os
import tensorflow as tf
import numpy as np
from configuration import IMG_HEIGHT, IMG_WIDTH, DATA_DIR


def load_data(data_dir, img_height, img_width, batch_size=None):
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
    
    return dataset, class_names