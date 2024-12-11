from tensorflow.keras import layers, models
from tensorflow.keras import regularizers


def build_cnn_model(num_classes, input_shape=(45, 45, 1)):
    model = models.Sequential([

        # Première couche de convolution
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Deuxième couche de convolution
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Deuxième couche de convolution
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),

        # Couche entièrement connectée
        layers.Flatten(), # Transforme la sortie 2D des convolutions en une seule dimension (vectorisation).

        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5), # Désactive aléatoirement 50% des neurones pour éviter le sur-apprentissage
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def model_summary(model):
    """Affiche un résumé du modèle"""
    model.summary()
