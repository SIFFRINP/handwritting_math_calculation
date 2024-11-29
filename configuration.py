import os


# * _ WINDOW ___________________________________________________________________
W_BASE_WIDTH  = 1000
W_BASE_HEIGHT = 400
W_NAME        = "handwritting_math_calculation" 
W_ICON_PATH   = os.path.join("icons", "icon.png")


# * _ TEXT _____________________________________________________________________
FONT_PATH      = os.path.join("fonts", "math.ttf")
MATH_TEXT_SIZE = 20
HINT_TEXT_SIZE = 15

# * _ COLORS ___________________________________________________________________
BACKGROUND_COLOR   = (40 , 44 , 52 )
DRAWING_AREA_COLOR = (255, 255, 255)
PEN_COLOR          = (0  , 0  , 0  )
TEXT_COLOR         = (0  , 0  , 0  )



if __name__ == "__main__": 
    print("\x1b[33m~[WARNING] This script is not meant to be executed.\x1b[0m"); 