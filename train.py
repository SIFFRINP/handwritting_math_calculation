import os
import tensorflow as tf
from scripts3.data_loader import load_data
from scripts3.preprocessing import preprocess_images, preprocess_dataset
from scripts3.model_builder import build_cnn_model, model_summary
from scripts3.model_trainer import compile_and_train, save_model
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Aller à la racine du projet
print(project_root)
data_dir = os.path.join(project_root, "handwritting_math_calculation", "data", "extracted_images_sort")
print(data_dir)


dataset, class_names = load_data(data_dir, img_height=45, img_width=45, batch_size=None)
# for image, label in dataset.take(5):
#     print(f"Image shape : {image.shape}, Étiquette : {label}")


# Spliter les données brutes
total_size = len(list(dataset))  # Calculer le nombre total d'exemples
train_size = int(0.8 * total_size)  # 80% pour l'entraînement
val_size = total_size - train_size


train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)


# print(f"Total d'exemples : {total_size}")
# print(f"Exemples dans train_dataset : {len(list(train_dataset))}")
# print(f"Exemples dans val_dataset : {len(list(val_dataset))}")


# Appliquer le prétraitement à chaque ensemble
train_ds = preprocess_dataset(train_dataset, batch_size=32)
val_ds = preprocess_dataset(val_dataset, batch_size=32)

# Afficher un aperçu pour valider le prétraitement

# counter = 0
# for image_batch, label_batch in train_ds.take(1):
#     counter += len(image_batch)
#     print(f"Shape du batch d'images (train) : {image_batch.shape}")
#     print(f"Shape du batch d'étiquettes (train) : {label_batch.shape}")
#     print(f"Valeurs min/max des images dans le batch : {tf.reduce_min(image_batch).numpy()}, {tf.reduce_max(image_batch).numpy()}")
#     for i in range(5):  # Visualiser 5 images
#         plt.imshow(image_batch[i].numpy().squeeze(), cmap='gray')
#         plt.title(f"Classe réelle : {class_names[label_batch[i].numpy()]}")
#         plt.axis('off')
#         plt.show()
# print(f"Nombre total d'images dans le batch traité : {counter}")


num_classes = len(class_names)
# print("num_classes : ",num_classes)
model = build_cnn_model(num_classes=num_classes, input_shape=(45, 45, 1))
print("Résumé du modèle :")
model_summary(model)


# # Entraîner le modèle
history = compile_and_train(model, train_ds, val_ds, epochs=15, learning_rate=0.001)

# # Afficher les métriques
# print(f"Historique des pertes : {history.history['loss']}")
# print(f"Historique des précisions : {history.history['accuracy']}")

# Sauvegarder le modèle
save_model(model, model_name="handwritten_math_calculator_model.keras", save_dir="models")
