import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from configuration import * 
from classes import Window
from functions import *
import pygame
import time


if __name__ == "__main__":
    # * _ INITIALISATIONS ______________________________________________________
    pygame.init()
    window = Window()


    # * _ MAIN LOOP ____________________________________________________________
    while window.get_running_state():
        window.update()

        # Pause the detection and prediction if the user is drawing. 
        if window.get_mouse_pressed():
            continue

        start_time = time.time()

        # Get the drawing area buffer and extract every digit from it. 
        img = window.get_draw_area_pxl_array(flip=True)
        bounding_boxes = get_symbol_bounding(img)
        regions = pixels_isolation(img, bounding_boxes)
        
        # Put the regions into the model. 
        # print(regions)
        window.set_error_text(f"time to detect: {(time.time() - start_time):.2f}secs.")
