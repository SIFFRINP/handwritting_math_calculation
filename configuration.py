import os


DEBUG = True

# * _ WINDOW ___________________________________________________________________
USE_ABS_SIZE    = False
W_BASE_WIDTH    = 1000
W_BASE_HEIGHT   = 400
W_NAME          = "handwritting_math_calculation" 
W_ICON_PATH     = os.path.join("icons", "icon.png")
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

# * _ PARSER ___________________________________________________________________
AVAILABLE_OPERATOR  = "+-*/"
PRIORITY_OPERATOR   = "*/"
END_EXPRESSION_CHAR = "="

# * _ AI _______________________________________________________________________
img_height = 45
img_width = 45
batch_size = 32
epochs = 20
learning_rate = 0.00015
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "handwritting_math_calculation", "data", "extracted_images_sort2")
image_path = os.path.join(project_root, "handwritting_math_calculation", "images", "NONN.png")
model_path = os.path.join(project_root, "handwritting_math_calculation", "models", "handwritten_math_calculator_model4.keras")
save_dir = "models"
model_name="handwritten_math_calculator_model14.keras"
# model_name="cnn_model_best.keras"
# class_names = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=']
class_names = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'div', 'x']

# * _ MODEL ____________________________________________________________________
MODEL_IMG_SIZE = 45 


if __name__ == "__main__": 
    print("\x1b[33m~[WARNING] This script is not meant to be executed.\x1b[0m"); 
