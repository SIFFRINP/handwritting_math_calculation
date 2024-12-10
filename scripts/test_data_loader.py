import os
import tensorflow as tf
from scripts.data_loader import load_data
from configuration import IMG_HEIGHT, IMG_WIDTH, DATA_DIR, PROJECT_ROOT

print("chemin data_dir : ", DATA_DIR)
data_bad = os.path.join(PROJECT_ROOT, "handwritting_math_calculation", "images")


def test_load_data():
    try:
        dataset, class_names = load_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, batch_size=None)
        assert isinstance(dataset, tf.data.Dataset), "Le dataset n'est pas du bon type"
        print("Le dataset est du bon type")
        assert isinstance(class_names, list), "Les noms de classes ne sont pas dans une liste"
        print("Les noms de classes sont bien dans une liste ")
        
        # Vérification des données
        for image_batch, label_batch in dataset.take(1):
            print(f"Image shape : {image_batch.shape}, Étiquette : {label_batch}")
            print(image_batch.shape[1:])

        total_images = tf.data.experimental.cardinality(dataset).numpy()
        print(f"Total d'images : {total_images}")
        class_counts = {class_name: len(os.listdir(os.path.join(DATA_DIR, class_name))) for class_name in os.listdir(DATA_DIR)}
        print(class_counts)
        
        # Vérification des classes
        expected_classes = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=']
        assert set(class_names) == set(expected_classes), f"Noms des classes incorrects : {class_names}"
        print("Test 1 réussi.")
        print(len(class_names), "classe_names : ", class_names)
    except Exception as e:
        print(f"Test 1 échoué : {e}")

    # Test 2 : Chemin invalide
    try:
        load_data("path/that/does/not/exist", IMG_HEIGHT, IMG_WIDTH)
        print("Test 2 échoué : Une exception était attendue.")
    except Exception:
        print("Test 2 réussi.")

    # Test 3 : Structure incorrecte
    try:
        dataset, class_names = load_data(data_bad, IMG_HEIGHT, IMG_WIDTH)
        assert len(class_names) > 0, "Aucune classe trouvée"
        print("Test 3 réussi.")
    except Exception as e:
        print(f"Test 3 échoué : {e}")

test_load_data()

