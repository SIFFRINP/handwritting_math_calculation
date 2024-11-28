import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
from configuration import * 


if __name__ == "__main__":

    # * _ WINDOW INIT __________________________________________________________
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('handwritting_math_calculation')


    running = True
    while running:

    # * _ MAIN LOOP ____________________________________________________________
        x, y = pygame.mouse.get_pos()
        
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.circle(screen, PEN_COLOR, (x, y), 5)
        pygame.display.flip()


    # * _ EVENT HANDLER ________________________________________________________
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

