from configuration import *
import pygame

class Window:

    def __init__(self):
        self.is_running = True
        self.width = W_BASE_WIDTH
        self.height = W_BASE_HEIGHT

        self.screen = pygame.display.set_mode((self.width, self.height))
        
        self.drawing_area = pygame.Surface((self.width, self.height - (self.height / 6)))
        self.drawing_area.fill(DRAWING_AREA_COLOR)
        
        self.texts = []

        set_window_attributes()
        return


    def render_texts(self): 
        for text in self.texts: 
            self.screen.blit(text.get_text_surface(), text.get_coords())


    def render(self): 
        self.screen.fill(BACKGROUND_COLOR)
        self.screen.blit(self.drawing_area, (0, 0))
        
        self.render_texts()
        return



    def draw(self): 
        mouse_state = pygame.mouse.get_pressed(num_buttons=3)

        if not mouse_state[0] and not mouse_state[2]:
            return
        
        mouse_coords = pygame.mouse.get_pos()
        pygame.draw.circle(self.drawing_area, PEN_COLOR, mouse_coords, 5)
        return

    def show(self): 
        self.event_handler()

        self.draw()

        self.render()
        pygame.display.flip()
        return



    def clear_drawing_area(self):
        self.drawing_area.fill(DRAWING_AREA_COLOR)
        return



    def event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.clear_drawing_area()
        return 



    # * _ GETTERS & SETTERS ____________________________________________________
    def get_running_state(self) -> bool: 
        return self.is_running



# * _ STATIC FUNCTIONS _________________________________________________________
def set_window_attributes(): 
    if os.path.isfile(W_ICON_PATH):
        w_icon = pygame.image.load(W_ICON_PATH)
        pygame.display.set_icon(w_icon)
    
    pygame.display.set_caption(W_NAME)
    return


if __name__ == "__main__": 
    print("\x1b[33m~[WARNING] This script is not meant to be executed.\x1b[0m"); 
