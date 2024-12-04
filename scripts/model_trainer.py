import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
from scripts.model_builder import build_cnn_model


def compile_and_train(model, train_ds, val_ds, epochs=20, learning_rate=0.001):
    # Préparation du modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("cnn_model_best.keras", save_best_only=True)
    ]

    # Entraînement
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history


# def plot_training_history(history, save_path="training_history.json"):
#     # Sauvegarder l'historique dans un fichier JSON
#     history_dict = history.history
#     with open(save_path, 'w') as f:
#         json.dump(history_dict, f)
#     # Ensuite, tu peux faire le tracé des courbes comme avant.
#     plt.plot(history.history['accuracy'], label='accuracy')
#     plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.ylim([0, 1])
#     plt.legend(loc='lower right')
#     plt.show()


def save_model(model, model_name="handwritten_math_calculator_model.keras", save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print(f"Modèle sauvegardé dans : {model_path}")





# A savoir sur les callbacks
# EarlyStopping :

# Arrête l'entraînement si la performance sur la validation (ex. perte ou précision) ne s'améliore plus après patience epochs consécutifs.
# Option restore_best_weights=True : recharge automatiquement les poids qui ont donné les meilleurs résultats.
# Exemple : Si l'entraînement dure 10 epochs mais que le modèle atteint son pic de performance à epoch 6, cette callback arrête l'entraînement à epoch 9 et recharge les poids de l'epoch 6.

# ModelCheckpoint :

# Sauvegarde le modèle après chaque epoch si la performance s’améliore.
# Option save_best_only=True : ne garde que le meilleur modèle (selon la perte de validation, par défaut).