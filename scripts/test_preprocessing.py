import os
import tensorflow as tf
from scripts.data_loader import load_data
from scripts.preprocessing import preprocess_images, preprocess_dataset, preprocess_image_main
from PIL import Image
import numpy as np
from configuration import DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

# __ INITIALISATION __________________


print(DATA_DIR)
dataset, class_names = load_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH)

total_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.8 * total_size)  # 80% pour l'entraînement
val_size = total_size - train_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Réduire la taille pour les tests (32 images par split)
test_subset = train_dataset.take(32)




# __ TEST POUR PREPROCESS_IMAGES ____________________

print("\n=== Test pour preprocess_images ===")
for image, label in test_subset.take(1):
    print(f"Image originale shape : {image.shape}, Label : {label}")
    image_processed, label_processed = preprocess_images(image, label)
    print(f"Image après traitement shape : {image_processed.shape}")
    print(f"Valeurs min/max avant normalisation : {image.numpy().min()} / {image.numpy().max()}")
    print(f"Valeurs min/max après normalisation : {image_processed.numpy().min()} / {image_processed.numpy().max()}")
    print(f"Label traité : {label_processed}")
    assert image_processed.shape == (45, 45, 1), "Erreur dans la forme de l'image prétraitée"
    assert image_processed.numpy().max() <= 1.0, "L'image n'est pas normalisée correctement"
    assert image_processed.numpy().min() >= 0.0, "L'image contient des valeurs négatives"
    print("Test preprocess_images : OK")


# __ TEST POUR PREPROCESS_DATASET ____________________

print("\n=== Test pour preprocess_dataset ===")
preprocessed_dataset = preprocess_dataset(test_subset, BATCH_SIZE)
for batch_images, batch_labels in preprocessed_dataset.take(1):
    print(f"Batch images shape : {batch_images.shape}, Batch labels : {batch_labels}")
    print(f"Valeurs min/max dans le batch : {batch_images.numpy().min()} / {batch_images.numpy().max()}")
    print("Exemple d'une image après traitement (valeurs):")
    print(batch_images[0].numpy())
    assert batch_images.shape[1:] == (45, 45, 1), "Erreur dans la forme des images dans le batch"
    assert len(batch_labels) == batch_images.shape[0], "Erreur dans la correspondance des labels"
    print("Test preprocess_dataset : OK")


# __ TEST POUR PREPROCESS_IMAGE_MAIN ____________________

print("\n=== Test pour preprocess_image_main ===")
# Créer une image de test
test_image = Image.fromarray(np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8))
processed_image = preprocess_image_main(test_image, IMG_HEIGHT, IMG_WIDTH)
print(f"Image après traitement shape : {processed_image.shape}")
print(f"Valeurs min/max après traitement : {processed_image.numpy().min()} / {processed_image.numpy().max()}")
print(f"Image après traitement (valeurs) :")
print(processed_image.numpy()[0, :, :, 0])  # Afficher les valeurs du canal unique (gris)
assert processed_image.shape == (1, 45, 45, 1), "Erreur dans la forme de l'image unique prétraitée"
assert processed_image.numpy().max() <= 1.0, "L'image unique n'est pas normalisée correctement"
assert processed_image.numpy().min() >= 0.0, "L'image unique contient des valeurs négatives"
print("Test preprocess_image_main : OK")
