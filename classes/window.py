from configuration import *
from datetime import datetime
from PIL import Image
import pygame
import numpy
import time
import cv2

from classes.text import Text


class Window:
    """
    Window class. 

    Method available: 
    - draw() .................... : Handle the drawing feature onto the draw 
                                    area. 
    - clear_draw_area() ......... : Clear the drawing area surface. 
    - render() .................. : Render every element of the window onto the 
                                    main surface. 
    - update() .................. : Update the window renderer.  
    - event_handler() ........... : Handle every event of the window. 
    - get_running_state() ....... : return the state of the program (False if 
                                    it should close, True otherwise). 
    - get_mouse_pressed() ........: Return True if the user is drawing, False
                                    otherwise. 
    - get_draw_area_pxl_array() . : Return a 3D numpy array containing every 
                                    pixel of the surface. 
    - set_result_text(str) ...... : Set the result text to a new value. 
    - set_error_text(str) ....... : Set the error text to a new value. 
    """

    # * Constructor of the Window class. 
    def __init__(self):
        self.is_running = True

        # Define the size of the window. 
        if USE_ABS_SIZE:
            self.win_w = W_BASE_WIDTH
            self.win_h = W_BASE_HEIGHT
        else: 
            screen_w = pygame.display.Info().current_w
            screen_h = pygame.display.Info().current_h
            self.win_w = screen_w - (screen_w / 3)
            self.win_h = screen_h - (screen_h / 3)
        
        # Create the main window surface. 
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        
        # Create the drawing area surface. 
        self.draw_area = pygame.Surface((self.win_w, self.win_h - CALCULUS_HEIGHT))
        self.draw_area.fill(DRAWING_AREA_COLOR)

        # Mouse state. 
        self.mouse_is_pressed = False
        self.last_saved_pen_pos = -1, -1
        self.last_input_ts = -1

        # Texts
        self.calculus_text = Text("~")
        self.error_text = Text("", PEN_COLOR)

        self.constant_texts = [
            Text("press [q] to clear the renderer. "), 
            Text("press [esc] to exit the program. "), 
        ]

        self.constant_texts_coords = [
            (5, 0), 
            (5, 17), 
        ]

        set_window_attributes()
        return


    # * Handle the drawing on the drawing area using the mouse. 
    def draw(self): 
        # Check if left or right mouse buttons are pressed. 
        if not self.mouse_is_pressed:
            self.last_saved_pen_pos = -1, -1
            return
        
        mouse_coords = pygame.mouse.get_pos()
        
        # Check if there is a last saved pen position. 
        if self.last_saved_pen_pos[0] > -1 or self.last_saved_pen_pos[1] > -1:
            pygame.draw.line(
                self.draw_area, 
                PEN_COLOR, 
                (mouse_coords), 
                self.last_saved_pen_pos, 
                PEN_WIDTH
            )

        # Draw a circle at the beginning to smooth out the pen. 
        else: 
            pygame.draw.circle(self.draw_area, PEN_COLOR, mouse_coords, 5) 

        self.last_saved_pen_pos = mouse_coords
        return


    # * Clear the drawing area. 
    def clear_draw_area(self):
        self.draw_area.fill(DRAWING_AREA_COLOR)
        return



    # * Render the window element onto the main surface. 
    def render(self): 
        self.screen.fill(BACKGROUND_COLOR)
        self.screen.blit(self.draw_area, (0, 0))

        # Get the text surface. 
        res_text_surf = self.calculus_text.get_text_surface(scale_height=CALCULUS_HEIGHT)
        err_text_surf = self.error_text.get_text_surface(scale_height=25) 


        # Render the result text on the screen. 
        self.screen.blit(res_text_surf, (CALCULUS_HEIGHT / 4, self.win_h - CALCULUS_HEIGHT + 2))
        self.screen.blit(err_text_surf, (5, self.win_h - CALCULUS_HEIGHT - 25))
        
        for coords, text in zip(self.constant_texts_coords, self.constant_texts):
            self.screen.blit(text.get_text_surface(scale_height=25), coords)
        
        return


    # * Render and update the full window. 
    def update(self): 
        self.event_handler()

        # 
        self.draw()
        self.render()
        pygame.display.flip()
        return


    # * Handle the exit and keyboard input. 
    def event_handler(self):

        # Update mouse pressed state. 
        mouse_state = pygame.mouse.get_pressed(num_buttons=3)
        if mouse_state[0] or mouse_state[2]: 
            self.mouse_is_pressed = True

        else: 
            self.mouse_is_pressed = False

        # Handle window event. 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            
            # Clear the drawing area surface. 
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.clear_draw_area()
                
                # Close the program. 
                if event.key == pygame.K_ESCAPE: 
                    self.is_running = False

                # Save the drawing on the screen. 
                if event.key == pygame.K_s: 
                    if (not os.path.isdir("saved_drawing")): 
                        print("~[ERROR] saved_drawing directory don't exist.")
                        return
                    
                    timestamp = datetime.timestamp(datetime.now())

                    img = Image.fromarray(self.get_draw_area_pxl_array(flip=True))
                    img.save(os.path.join("saved_drawing", f"drawing_{timestamp}.png"))

        return 


    # * _ GETTERS & SETTERS ____________________________________________________
    def get_running_state(self) -> bool: 
        return self.is_running
    

    def get_mouse_pressed(self) -> bool: 
        if (self.mouse_is_pressed): 
            self.last_input_ts = time.time()
            return True
        
        if (time.time() - self.last_input_ts < DETECTION_WAIT_TIME):
            return True
         
        return False


    def get_draw_area_pxl_array(self, flip: bool = False) -> numpy.ndarray:
        buf = pygame.surfarray.array3d(self.draw_area)
        buf = cv2.cvtColor(buf, cv2.COLOR_BGR2GRAY)

        # Flip the height and width. 
        if flip: 
            return buf.swapaxes(0, 1)

        return buf


    def set_result_text(self, new_text: str): 
        self.calculus_text.set_text("~ " + new_text)
        return 
    

    def set_error_text(self, new_text: str): 
        self.error_text.set_text("~" + new_text)
        return 

# * _ STATIC FUNCTIONS _________________________________________________________
def set_window_attributes(): 
    if os.path.isfile(W_ICON_PATH):
        w_icon = pygame.image.load(W_ICON_PATH)
        pygame.display.set_icon(w_icon)
    
    pygame.display.set_caption(W_NAME)
    return


if __name__ == "__main__": 
    print("\x1b[33m~[WARNING] This script is not meant to be executed.\x1b[0m"); 
