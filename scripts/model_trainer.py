import tensorflow as tf
import os
from scripts.model_builder import build_cnn_model
from scripts.data_loader import preprocess_images, load_data

def compile_and_train(model, train_ds, val_ds, epochs=10, learning_rate=0.001):
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


def save_model(model, model_name="handwritten_math_calculator_model", save_dir="models"):
    """Sauvegarde un modèle dans le répertoire spécifié."""
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