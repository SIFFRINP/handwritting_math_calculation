from configuration import *
import pygame


class Text:
    """
    Text class. 

    Method available: 
    - get_text_surface() .... : Return the generated surface to blit onto 
                                another surface. 
    - set_text(str) ......... : Set the text to a new value. 
    """

    # * Constructor of the Text class. 
    def __init__(self, text, color=TEXT_COLOR):
        self.font = pygame.font.Font(FONT_PATH, FONT_SIZE)
        self.color = color
        self.text = text
        return


    # * _ GETTERS & SETTERS ____________________________________________________
    def get_text_surface(self, scale_height=None):
        text_surface = self.font.render(self.text, True, self.color)

        if (scale_height is not None): 
            # Resize it in height, keeping the aspect ratio. 
            text_w = text_surface.get_width()
            text_h = text_surface.get_height()
            aspect_ratio = text_w / text_h
            
            new_h = scale_height
            new_w = new_h * aspect_ratio
            
            # Resize the surface. 
            text_surface = pygame.transform.smoothscale(text_surface, (new_w, new_h))

        return text_surface

    def set_text(self, new_text: str): 
        self.text = new_text
        return
    

    # * _ METHOD OVERRIDES _____________________________________________________
    def __str__(self):
        return f"~[TEXT] \"{self.text}\" at [{self.x};{self.y}]."
