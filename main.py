import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
from configuration import * 
from classes.window import Window


if __name__ == "__main__":
    pygame.init()

    # * _ INITIALISATIONS ______________________________________________________
    window = Window()

    while window.get_running_state():
    # * _ MAIN LOOP ____________________________________________________________
        window.show()