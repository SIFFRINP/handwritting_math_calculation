import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
from functions import *
from configuration import * 
from classes import Window


INSTRUCTION = "8+4*10-3/2="
RESULT      = 46.5


if __name__ == "__main__":
    print("BASE INSTRUCTION: ")
    print(f"\t~ inst: {INSTRUCTION}")

    numbers, operators = separate_instructions(INSTRUCTION)
    print("\nPARSING RESULT: ")
    print(f"\t~ nb_parsing: {numbers}")
    print(f"\t~ op_parsing: {operators}")

    result = perform_calc(numbers, operators)
    print("\nCALCULATION RESULT: ")
    print(f"\t= {result} ?= {RESULT} | {"✅" if (RESULT == result) else "❌"}")

    print("\nFINAL STRING RESULT: ")
    print(f"\t~ {INSTRUCTION}{result}")


    # * _ INITIALISATIONS ______________________________________________________
    pygame.init()
    window = Window()
    window.set_result_text(f"{INSTRUCTION}{result}")


    # * _ MAIN LOOP ____________________________________________________________
    while window.get_running_state():
        try: 
            window.update()

        except KeyboardInterrupt: 
            print("\x1b[1m\x1b[32mGoodbye :)\x1b[0m\n")
            break
        