import os

# DEBUG LEVELS: 
# - 1: Only print statements. 
# - 2: Images processing debug. 
DEBUG = 1


def DEBUG_PRINT(message: str): 
    if DEBUG: 
        print(message)

# * _ WINDOW ___________________________________________________________________
USE_ABS_SIZE     = False
W_BASE_WIDTH     = 1000
W_BASE_HEIGHT    = 400
W_NAME           = "handwritting_math_calculation" 
W_ICON_PATH      = os.path.join("icons", "icon.png")
CALCULUS_HEIGHT  = 65
TEXT_Y_SPACEMENT = 17

# * _ TEXT _____________________________________________________________________
FONT_PATH = os.path.join("fonts", "math.ttf")
FONT_SIZE = 100

# * _ DRAW AREA ________________________________________________________________
PEN_WIDTH           = 3
DETECTION_WAIT_TIME = 1

# * _ COLORS ___________________________________________________________________
BACKGROUND_COLOR   = (211, 211, 211)
DRAWING_AREA_COLOR = (255, 255, 255)
PEN_COLOR          = (0  , 0  , 0  )
TEXT_COLOR         = (0  , 0  , 0  )
ERROR_COLOR        = (156, 69 , 79 )

# * _ PARSER ___________________________________________________________________
AVAILABLE_OPERATOR  = "+-×÷"
PRIORITY_OPERATOR   = "×÷"
END_EXPRESSION_CHAR = "="


# * _ AI PARAMETERS ____________________________________________________________
IMG_HEIGHT    = 45
IMG_WIDTH     = 45
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 0.00015
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(PROJECT_ROOT, "handwritting_math_calculation", "data", "symbols")
IMAGE_PATH    = os.path.join(PROJECT_ROOT, "handwritting_math_calculation", "images", "NONN.png")
MODEL_NAME    = "handwritten_math_calculator_model14.keras"
MODEL_PATH    = os.path.join(PROJECT_ROOT, "handwritting_math_calculation", "models", MODEL_NAME)
SAVE_DIR      = "models"
CLASS_NAMES   = ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", '÷', '×']

# * _ ANSI COLOR _______________________________________________________________
WARNING = "\x1b[33m"
SUCCESS = "\x1b[32m"
INFO    = "\x1b[1m"

CLEAR_STYLE = "\x1b[0m"


if __name__ == "__main__": 
    print(f"{WARNING}~[WARNING] This script is not meant to be executed.{CLEAR_STYLE}"); 
