import os

img_height = 45
img_width = 45
batch_size = 32
epochs = 20
learning_rate = 0.001
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data", "extracted_images_sort")
image_path = os.path.join(project_root, "images", "OUII.png")
model_path = os.path.join(project_root, "models", "handwritten_math_calculator_model4.keras")
save_dir = "models"
model_name="handwritten_math_calculator_model4.keras"
class_names = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=']
