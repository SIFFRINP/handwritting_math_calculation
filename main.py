import os
from configuration import * 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
DEBUG_PRINT(f"{INFO}~[INFO] Importing tensorflow, this can take a while...{CLEAR_STYLE}")

import time
import pygame
import numpy as np
import tensorflow as tf

from scripts import *
from classes import Window
from scripts.preprocessing import preprocess_image_main



if __name__ == "__main__":
    # * _ INITIALISATIONS ______________________________________________________
    timer = time.time()
    
    DEBUG_PRINT(f"{INFO}~[INFO] Loading model from: {MODEL_NAME}.")
    model = tf.keras.models.load_model(MODEL_PATH)
    DEBUG_PRINT(f"{INFO}~[INFO] Model loaded in {time.time() - timer:.2f}sec {SUCCESS}✓{CLEAR_STYLE}")
   
    pygame.init()
    window = Window()


    # * _ MAIN LOOP ____________________________________________________________
    instruction = ""
    as_predicted = False

    while window.get_running_state():
        window.update()

        # Pause the detection and prediction if the user is drawing. 
        if window.get_mouse_pressed(): 
            as_predicted = False
            continue

        # If the currently drawn operation as already been predicted, don't 
        # predict it again. 
        if (as_predicted): 
            continue

        # Otherwise, detect, predict and calculate. 
        as_predicted = True
        timer = time.time()

        # Get the drawing area buffer and extract every digit from it. 
        img = window.get_draw_area_pxl_array(flip=True)
        bounding_boxes = get_symbol_bounding(img)

        # Sort every bounding box according to their x axis so it predict from 
        # left to right. 
        bounding_boxes.sort(key=lambda elt: elt[0])
        regions = pixels_isolation(img, bounding_boxes)
        
        # Put the regions into the model. 
        instruction = ""

        for region in regions:
            image_clean = preprocess_image_main(region, IMG_HEIGHT, IMG_WIDTH)

            # Predict the drawing. 
            predictions = model.predict(image_clean, verbose=0)
            predicted_class = CLASS_NAMES[np.argmax(predictions, axis=1)[0]]
    
            # Append the predicted symbol to the instruction. 
            instruction += predicted_class

        # Calculate the predicted instruction. 
        numbers, operators = separate_instructions(instruction)
        result = perform_calc(numbers, operators)
        
        # And display the result. 
        window.set_result_text(f"{instruction_format(instruction)}{result}")
        
        if DEBUG: 
            window.set_error_text(f"time to detect: {(time.time() - timer):.2f}secs.")

    print(f"{INFO}GOODBYE :).")
