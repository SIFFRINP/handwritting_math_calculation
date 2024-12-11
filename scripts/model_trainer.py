import tensorflow as tf
import os
import matplotlib.pyplot as plt
from scripts.model_builder import build_cnn_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from configuration import epochs, learning_rate, model_name, save_dir


def compile_and_train(model, train_ds, val_ds, epochs, learning_rate, class_weight=None):
    # Préparation du modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=4, 
            restore_best_weights=True, 
            monitor='val_loss',  # Monitor la perte de validation (critère standard)
            mode='min'  # Chercher à minimiser la perte
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "cnn_model_best.keras", 
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",  # Métrique à surveiller
            factor=0.5,          # Divise le learning_rate par 2
            patience=4,          # Réduit après 2 époques sans amélioration
            verbose=1            # Affiche un message lors de la réduction
        )
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


def save_model(model, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print(f"Modèle sauvegardé dans : {model_path}")



def evaluate_metrics_per_class(model, val_ds, class_names, show_errors=False):
    # Générer les prédictions
    y_true = []
    y_pred = []
    images = []

    for images_batch, labels in val_ds:
        predictions = model.predict(images_batch)
        y_true.extend(labels.numpy())  # Les vraies étiquettes
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())  # Prédictions du modèle
        images.extend(images_batch.numpy())

    # Générer le rapport par classe
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names
    )
    print("=== Rapport de classification par classe ===")
    print(report)

    # Analyser les erreurs si demandé
    if show_errors:
        errors = [(img, true, pred) for img, true, pred in zip(images, y_true, y_pred) if true != pred]

        print(f"Nombre d'erreurs : {len(errors)}")
        for i, (img, true, pred) in enumerate(errors[:10]):  # Limite à 10 erreurs affichées
            plt.imshow(img.squeeze(), cmap="gray")
            plt.title(f"Vrai : {class_names[true]}, Prédit : {class_names[pred]}")
            plt.show()



def plot_confusion_matrix(model, val_ds, class_names):
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    plt.show()
