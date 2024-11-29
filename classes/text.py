import pygame

from configuration import *


class Text:
    """
    Text class. 

    Method available: 
    - get_text_surface() .... : Return the generated surface to blit onto 
                                another surface. 
    - set_text(str) ......... : Set the text to a new value. 
    """

    # * Constructor of the Text class. 
    def __init__(self, text, font_size):
        self.font = pygame.font.Font(FONT_PATH, font_size)
        self.text = text
        return


    # * _ GETTERS & SETTERS ____________________________________________________
    def get_text_surface(self):
        return self.font.render(self.text, True, TEXT_COLOR)

    def set_text(self, new_text: str): 
        self.text = new_text
        return
    

    # * _ METHOD OVERRIDES _____________________________________________________
    def __str__(self):
        return f"~[TEXT] \"{self.text}\" at [{self.x};{self.y}]."
