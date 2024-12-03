from scripts.data_loader import preprocess_images, load_data
from scripts.model_builder import build_cnn_model, model_summary
from scripts.model_trainer import compile_and_train, save_model
import tensorflow as tf
import os


# Charger les données
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Aller à la racine du projet
print(project_root)
data_dir = os.path.join(project_root, "handwritting_math_calculation", "data", "extracted_images")
print(data_dir)

train_ds, val_ds, test_ds, class_names = load_data(data_dir, img_height=45, img_width=45)


# Prétraitement
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(preprocess_images, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess_images, num_parallel_calls=AUTOTUNE)

# Construire le modèle
input_shape = (45, 45, 1)
num_classes = len(class_names)
model = build_cnn_model(num_classes, input_shape=input_shape)
model_summary(model)

# Compiler et entraîner
history = compile_and_train(model, train_ds, val_ds, epochs=10)
print(f"Précision finale sur validation : {history.history['val_accuracy'][-1]}")


# Sauvegarder le modèle
save_model(model)