import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scripts.data_loader import load_data
from scripts.preprocessing import preprocess_dataset
from scripts.model_builder import build_cnn_model
from scripts.model_trainer import compile_and_train, save_model
from configuration import EPOCHS, learning_rate, DATA_DIR, IMG_HEIGHT, IMG_WIDTH



# __ INITIALISATION __________________

print(DATA_DIR)
dataset, class_names = load_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH)

total_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.8 * total_size)  # 80% pour l'entraînement
val_size = total_size - train_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)
assert len(list(train_dataset)) == train_size, "Le dataset d'entraînement n'a pas la bonne taille"
print("Le dataset d'entraînement a la bonne taille")
assert len(list(val_dataset)) == val_size, "Le dataset de validation n'a pas la bonne taille"
print("Le dataset de validation a la bonne taille")


train_dataset = dataset.take(20)  # On limite à 20 images pour un test rapide
val_dataset = dataset.skip(20)
train_ds = preprocess_dataset(train_dataset, batch_size=2)
# train_ds = preprocess_dataset(train_dataset, batch_size=2, augment=True)
val_ds = preprocess_dataset(val_dataset, batch_size=2)
# val_ds = preprocess_dataset(train_dataset, batch_size=2, augment=False)


input_shape = (45, 45, 1)
num_classes = len(class_names)
print("num_classes : ", num_classes)
model = build_cnn_model(num_classes=num_classes, input_shape=input_shape)


# __ TESTS __________________


# Afficher les détails du modèle
print("\n=== Vérification du modèle ===")
print(f"Forme d'entrée : {model.input_shape}")
print(f"Forme de sortie : {model.output_shape}")
print(f"Nombre total de couches : {len(model.layers)}")
for i, layer in enumerate(model.layers):
    print(f"Couche {i} - Type : {type(layer).__name__}")
    if hasattr(layer, 'output_shape'):
        print(f"  - Forme de sortie : {layer.output_shape}")
        

# Calcul des poids de classes
class_counts = {'+': 25112, '-': 33997, '0': 6914, '1': 26520, '2': 26141, 
                '3': 10909, '4': 7396, '5': 3545, '6': 3118, '7': 2909, 
                '8': 3068, '9': 3737, '=': 13104}
total_images = sum(class_counts.values())
class_weights = {i: total_images / count for i, count in enumerate(class_counts.values())}
print("Poids des classes : ", class_weights)


# Étape 5 : Compilation et entraînement
print("\n=== Compilation et entraînement ===")
history = compile_and_train(
    model,
    train_ds,
    val_ds,
    epochs=2,
    learning_rate=learning_rate,
    class_weight=class_weights
)


# Vérification des résultats
print("\n=== Résultats de l'entraînement ===")
print(f"Historique des pertes : {history.history['loss']}")
assert history.history['loss'][-1] < history.history['loss'][0], "La perte ne diminue pas"
print("La perte diminue")
print(f"Historique de l'accuracy : {history.history['accuracy']}")
assert history.history['accuracy'][-1] > history.history['accuracy'][0], "L'accuracy n'augmente pas"
print("L'accuracy augmente")



plt.plot(history.history['loss'], label='Perte - Entraînement')
plt.plot(history.history['val_loss'], label='Perte - Validation')
plt.legend()
plt.title('Courbe de perte')
plt.show()

plt.plot(history.history['accuracy'], label='Accuracy - Entraînement')
plt.plot(history.history['val_accuracy'], label='Accuracy - Validation')
plt.legend()
plt.title('Courbe d\'accuracy')
plt.show()

# Test sur plusieurs batches du dataset de validation
print("\n=== Test des prédictions sur plusieurs batches ===")
batch_count = 0
for batch_images, batch_labels in val_ds.take(5):  # Par exemple, teste sur 5 batches
    predictions = model(batch_images)
    print(f"\nBatch {batch_count + 1} :")
    print(f"Forme des prédictions : {predictions.shape}")
    print(f"Probabilités pour la première image : {predictions.numpy()[0]}")
    print(f"Classe prédite pour la première image : {np.argmax(predictions.numpy()[0])}")
    print(f"Classe réelle pour la première image : {batch_labels.numpy()[0]}")
    batch_count += 1

print("=== Tests de model_trainer.py terminés avec succès ===")


