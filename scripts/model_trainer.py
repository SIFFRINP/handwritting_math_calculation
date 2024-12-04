import tensorflow as tf
import os
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


def save_model(model, model_name="handwritten_math_calculator_model.keras", save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print(f"Modèle sauvegardé dans : {model_path}")