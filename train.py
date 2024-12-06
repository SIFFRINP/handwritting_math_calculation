import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scripts.data_loader import load_data
from scripts.preprocessing import preprocess_dataset
from scripts.model_builder import build_cnn_model, model_summary
from scripts.model_trainer import compile_and_train, save_model, evaluate_metrics_per_class


# __ INITIALISATION __________________

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Aller à la racine du projet
print(project_root)
data_dir = os.path.join(project_root, "handwritting_math_calculation", "data", "extracted_images_sort")
print(data_dir)
dataset, class_names = load_data(data_dir, img_height=45, img_width=45)


total_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.8 * total_size)  # 80% pour l'entraînement
val_size = total_size - train_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)


assert tf.data.experimental.cardinality(train_dataset).numpy() == train_size, "Le dataset d'entraînement n'a pas la bonne taille"
print("Le dataset d'entraînement a la bonne taille")
assert tf.data.experimental.cardinality(val_dataset).numpy() == val_size, "Le dataset de validation n'a pas la bonne taille"
print("Le dataset de validation a la bonne taille")


train_ds = preprocess_dataset(train_dataset, batch_size=32)
val_ds = preprocess_dataset(val_dataset, batch_size=32)

input_shape = (45, 45, 1)
num_classes = len(class_names)
print("num_classes : ", num_classes)
model = build_cnn_model(num_classes=num_classes, input_shape=input_shape)


print("Résumé du modèle :")
model_summary(model)


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
    epochs=20,
    learning_rate=0.001,
    class_weight=class_weights
)


# Vérifier les métriques par classe
print("\n=== Vérification des métriques par classe ===")
evaluate_metrics_per_class(model, val_ds, class_names)


# Vérification des résultats
print("\n=== Résultats de l'entraînement ===")
print(f"Historique des pertes : {history.history['loss']}")
assert history.history['loss'][-1] < history.history['loss'][0], "La perte ne diminue pas"
print("La perte diminue")
print(f"Historique de l'accuracy : {history.history['accuracy']}")
assert history.history['accuracy'][-1] > history.history['accuracy'][0], "L'accuracy n'augmente pas"
print("L'accuracy augmente")



# Récupérer les données d'entraînement et validation
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']


# Afficher les métriques sur chaque époque
print("=== Évolution des métriques sur les époques ===")
for i in range(len(train_loss)):
    print(f"Époque {i+1}:")
    print(f"  Perte - Entraînement : {train_loss[i]:.4f}, Validation : {val_loss[i]:.4f}")
    print(f"  Accuracy - Entraînement : {train_accuracy[i]:.4f}, Validation : {val_accuracy[i]:.4f}")


plt.plot(train_loss, label='Perte - Entraînement')
plt.plot(val_loss, label='Perte - Validation')
plt.legend()
plt.title('Courbe de perte')
plt.show()

plt.plot(train_accuracy, label='Accuracy - Entraînement')
plt.plot(val_accuracy, label='Accuracy - Validation')
plt.legend()
plt.title('Courbe d\'accuracy')
plt.show()


print("=== Tests de model_trainer.py terminés avec succès ===")


# Sauvegarder le modèle
save_model(model, model_name="handwritten_math_calculator_model3.keras", save_dir="models")