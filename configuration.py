import os


# * _ WINDOW ___________________________________________________________________
USE_ABS_SIZE  = False
W_BASE_WIDTH  = 1000
W_BASE_HEIGHT = 400
W_NAME        = "handwritting_math_calculation" 
W_ICON_PATH   = os.path.join("icons", "icon.png")

CALCULUS_HEIGHT = 65


# * _ DRAW AREA ________________________________________________________________
PEN_WIDTH = 10

# * _ TEXT _____________________________________________________________________
FONT_PATH = os.path.join("fonts", "math.ttf")
FONT_SIZE = 100

# * _ COLORS ___________________________________________________________________
BACKGROUND_COLOR   = (211, 211, 211)
DRAWING_AREA_COLOR = (255, 255, 255)
PEN_COLOR          = (0  , 0  , 0  )
TEXT_COLOR         = (0  , 0  , 0  )
ERROR_COLOR        = (156, 69 , 79 )



if __name__ == "__main__": 
    print("\x1b[33m~[WARNING] This script is not meant to be executed.\x1b[0m"); 