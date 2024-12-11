import tensorflow as tf
import matplotlib.pyplot as plt
from scripts.data_loader import load_data
from scripts.preprocessing import preprocess_dataset, print_dataset_distribution, downsample_classes, augment_classes, data_augmentation
from scripts.model_builder import build_cnn_model, model_summary
from scripts.model_trainer import compile_and_train, save_model, evaluate_metrics_per_class, plot_confusion_matrix
from configuration import data_dir, img_width, img_height, batch_size, epochs, learning_rate, save_dir, model_name


# __ INITIALISATION __________________

print("Avant downsampling et augmentation :")
print_dataset_distribution(data_dir)

downsample_classes(data_dir, target_size=10000)
augment_classes(data_dir, target_size=10000, augment_function=data_augmentation)

print("Après downsampling et augmentation :")
print_dataset_distribution(data_dir)


print(data_dir)
dataset, class_names = load_data(data_dir, img_height, img_width)


total_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.8 * total_size)  # 80% pour l'entraînement
val_size = total_size - train_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)


assert tf.data.experimental.cardinality(train_dataset).numpy() == train_size, "Le dataset d'entraînement n'a pas la bonne taille"
print("Le dataset d'entraînement a la bonne taille")
assert tf.data.experimental.cardinality(val_dataset).numpy() == val_size, "Le dataset de validation n'a pas la bonne taille"
print("Le dataset de validation a la bonne taille")


train_ds = preprocess_dataset(train_dataset, batch_size, augment=True)
val_ds = preprocess_dataset(val_dataset, batch_size, augment=False)


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



# Étape 5 : Compilation et entraînement
print("\n=== Compilation et entraînement ===")
history = compile_and_train(
    model,
    train_ds,
    val_ds,
    epochs=epochs,
    learning_rate=learning_rate
)


# Vérifier les métriques par classe
print("\n=== Vérification des métriques par classe ===")
# evaluate_metrics_per_class(model, val_ds, class_names)
evaluate_metrics_per_class(model, val_ds, class_names, show_errors=True)


# Observer la confusion matrix
plot_confusion_matrix(model, val_ds, class_names)


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
save_model(model, model_name, save_dir)