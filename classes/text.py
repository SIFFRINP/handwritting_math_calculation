import pygame

from ..configuration import *


class Text:

    def __init__(self, text, font_size, x, y):
        self.font = pygame.font.Font(FONT_PATH, font_size)
        self.text = text
        self.x = x
        self.y = y


    def get_text_surface(self):
        return self.font.render(self.text, True, TEXT_COLOR)
    

    # * _ GETTERS & SETTERS ____________________________________________________
    def set_text(self, new_text): 
        self.text = new_text
        return
    

    def get_coords(self): 
        return self.x, self.y
    

    # * _ METHOD OVERRIDES _____________________________________________________
    def __str__(self):
        return f"~[TEXT] \"{self.text}\" at [{self.x};{self.y}]."
