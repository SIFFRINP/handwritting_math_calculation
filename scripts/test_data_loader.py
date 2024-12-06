import os
import tensorflow as tf
from data_loader import load_data

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Aller à la racine du projet
print(project_root)
data_dir = os.path.join(project_root, "handwritting_math_calculation", "data", "extracted_images_sort")
print(data_dir)
data_bad = os.path.join(project_root, "handwritting_math_calculation", "images")


def test_load_data():
    try:
        dataset, class_names = load_data(data_dir, img_height=45, img_width=45, batch_size=None)
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
        class_counts = {class_name: len(os.listdir(os.path.join(data_dir, class_name))) for class_name in os.listdir(data_dir)}
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
        load_data("path/that/does/not/exist", img_height=45, img_width=45)
        print("Test 2 échoué : Une exception était attendue.")
    except Exception:
        print("Test 2 réussi.")

    # Test 3 : Structure incorrecte
    try:
        dataset, class_names = load_data(data_bad, img_height=45, img_width=45)
        assert len(class_names) > 0, "Aucune classe trouvée"
        print("Test 3 réussi.")
    except Exception as e:
        print(f"Test 3 échoué : {e}")

test_load_data()

