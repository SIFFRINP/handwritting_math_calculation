import os
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# import pygame
# from functions import *
# from configuration import * 
# from classes import Window
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from scripts.preprocessing import preprocess_images, preprocess_dataset, preprocess_image_main
import matplotlib.pyplot as plt
from scripts.config import image_path, model_path, class_names, img_height, img_width


# INSTRUCTION = "8+4*10-3/2="
# RESULT      = 46.5


if __name__ == "__main__":
    # print("BASE INSTRUCTION: ")
    # print(f"\t~ inst: {INSTRUCTION}")

    # numbers, operators = separate_instructions(INSTRUCTION)
    # print("\nPARSING RESULT: ")
    # print(f"\t~ nb_parsing: {numbers}")
    # print(f"\t~ op_parsing: {operators}")

    # result = perform_calc(numbers, operators)
    # print("\nCALCULATION RESULT: ")
    # print(f"\t= {result} ?= {RESULT} | {"✅" if (RESULT == result) else "❌"}")

    # print("\nFINAL STRING RESULT: ")
    # print(f"\t~ {INSTRUCTION}{result}")


    # * _ INITIALISATIONS ______________________________________________________
    # pygame.init()
    # window = Window()
    # window.set_result_text(f"{INSTRUCTION}{result}")

    # Chargement du modèle
    model = tf.keras.models.load_model(model_path)
    print(f"Modèle chargé depuis : {model_path}")

    # # Chargement de l'image
    # image = load_img(image_path)
    # plt.imshow(image)
    # plt.axis('off')
    # plt.title("image avant pré-traitement")
    # plt.show()

    # image_clean = preprocess_image_main(image, 45, 45)

    input_array = np.random.randint(0, 255, size=(45, 45, 3), dtype=np.uint8)  # Exemple de tableau
    image_clean = preprocess_image_main(input_array, img_height, img_width)


    # _ TEST __________________________________________
    # plt.imshow(image_clean.numpy().squeeze(), cmap='gray')
    # plt.title("image après pré-traitement")
    # plt.axis('off')
    # plt.show()

    # print("Forme de l'image avant prédiction :", image_clean.shape)

    # Prédiction de l'image
    predictions = model.predict(image_clean)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    predicted_class = class_names[predicted_class_index]

    print(f"Le symbole prédit est : {predicted_class} avec une confiance de {confidence:.2f}%")

    # # * _ MAIN LOOP ____________________________________________________________
    # while window.get_running_state():
    #     try: 
    #         window.update()

    #     except KeyboardInterrupt: 
    #         print("\x1b[1m\x1b[32mGoodbye :)\x1b[0m\n")
    #         break
