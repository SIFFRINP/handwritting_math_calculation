import os
import tensorflow as tf
from scripts.data_loader import load_data
from scripts.preprocessing import preprocess_dataset
from scripts.model_builder import build_cnn_model, model_summary
from scripts.config import data_dir, img_width, img_height

# __ INITIALISATION __________________

print(data_dir)
dataset, class_names = load_data(data_dir, img_height, img_width)

total_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.8 * total_size)  # 80% pour l'entraînement
val_size = total_size - train_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

train_dataset = dataset.take(20)  # On limite à 20 images pour un test rapide
# train_ds = preprocess_dataset(train_dataset, batch_size=2)
train_ds = preprocess_dataset(train_dataset, batch_size=2, augment=True)


input_shape = (45, 45, 1)
num_classes = len(class_names)
print("num_classes : ", num_classes)
model = build_cnn_model(num_classes=num_classes, input_shape=input_shape)


# __ TESTS __________________


# Imprimer le résumé du modèle
print("\nRésumé du modèle :")
model_summary(model)
# print("Nombre total de paramètres :", model.count_params())



# === Étape 4 : Tester le modèle avec les données prétraitées ===
print("\n=== Test avec un batch réel des données prétraitées ===")
for batch_images, batch_labels in train_ds.take(1):
    print(f"Forme du batch d'entrée : {batch_images.shape}")
    output = model(batch_images)
    print(f"Forme de sortie : {output.shape}")
    print(f"Sortie attendue : ({2}, {num_classes}) (batch avec des probabilités pour chaque classe)")


# Afficher les détails des couches
print("\n=== Détails des couches ===")
for i, layer in enumerate(model.layers):
    print(f"Couche {i} - Type : {type(layer).__name__}")
    print(f"  - Configuration : {layer.get_config()}")
    if hasattr(layer, 'output_shape'):
        print(f"  - Forme de sortie : {layer.output_shape}")
    print("")

# === Étape 5 : Vérifications automatiques ===
print("\n=== Vérifications automatiques ===")
assert model.input_shape == (None, *input_shape), f"Forme d'entrée incorrecte : {model.input_shape}"
assert model.output_shape == (None, num_classes), f"Forme de sortie incorrecte : {model.output_shape}"

expected_layers = [tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization,
                   tf.keras.layers.MaxPooling2D, tf.keras.layers.Conv2D,
                   tf.keras.layers.BatchNormalization, tf.keras.layers.MaxPooling2D,
                   tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization,
                   tf.keras.layers.Flatten, tf.keras.layers.Dense,
                   tf.keras.layers.Dropout, tf.keras.layers.Dense]
actual_layers = [type(layer) for layer in model.layers]
assert actual_layers == expected_layers, f"Architecture incorrecte : {actual_layers}"

print("Test build_cnn_model avec pipeline complet : OK\n")


