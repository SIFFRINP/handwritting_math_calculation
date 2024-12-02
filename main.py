import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
from configuration import * 
from classes import Window


if __name__ == "__main__":
    # * _ INITIALISATIONS ______________________________________________________
    pygame.init()
    window = Window()


    # * _ MAIN LOOP ____________________________________________________________
    while window.get_running_state():
        try: 
            window.update()

        except KeyboardInterrupt: 
            print("\x1b[1m\x1b[32mGoodbye :)\x1b[0m\n")
            break
        