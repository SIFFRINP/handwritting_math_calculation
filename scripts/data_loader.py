import tensorflow as tf
import os

# Charger les données
def load_data(data_dir, img_height=45, img_width=45, batch_size=32, test_split=0.1):
    # Vérification du chemin
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Chemin invalide ou dossier introuvable : {data_dir}")

    print(f"Chemin des données valide : {data_dir}")

    # Chargement des données. Je charge tout le dataset d'un coup sans aucun split
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Création des splits
    total_size = len(dataset) 
    test_size = int(total_size * test_split) # Split pour les tests
    train_val_size = total_size - test_size # Split pour l'entrapinement + validation

    train_val_ds = dataset.take(train_val_size)
    test_ds = dataset.skip(train_val_size)

    val_split = 0.2
    train_size = int(train_val_size * (1 - val_split))
    val_size = train_val_size - train_size

    train_ds = train_val_ds.take(train_size)
    val_ds = train_val_ds.skip(train_size)

    class_names = dataset.class_names
    print(f"Classes trouvées : {class_names}")

    return train_ds, val_ds, test_ds, class_names

def preprocess_images(image, label):
    # Conversion en niveaux de gris et normalisation
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


if __name__ == "__train__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Aller à la racine du projet
    data_dir = os.path.join(project_root, "data", "extracted_images")
    load_data(data_dir, img_height=45, img_width=45)
