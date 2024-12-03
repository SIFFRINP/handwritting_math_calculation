from configuration import * 
import numpy as np
import cv2


def get_symbol_bounding(img: np.ndarray) -> list: 
    """
    Get the coordinates of each symbol by doing a contour detection on the image. 

    :param img: numpy array of the drawing area.  
    :return: a list that contains every bounding box of every symbol on the 
             drawing area. 
    """

    bounding_boxes = []

    # Convert img to grayscale and threshold it for the contour detection. 
    _, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Perform the contour detection. 
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Append every contour to the bounding_boxes list. 
        bounding_boxes.append(cv2.boundingRect(contour))
    
    return bounding_boxes


def pixels_isolation(img: np.ndarray, bounding_boxes: list): 
    """
    Take an image and bounding boxes and extract pixels from the image contained
    in those bounding boxes and put them into an array. 

    :param img: the image where pixels need to be captured. 
    :param bounding_boxes: every bounding boxes. 
    :return: an array of region of the base images. 
    """

    # Create the numpy array. 
    region_count = len(bounding_boxes)
    regions = np.zeros((region_count, MODEL_IMG_SIZE, MODEL_IMG_SIZE), dtype=np.uint8)

    # Retrieve each region from the images. 
    for i, box in enumerate(bounding_boxes): 
        x, y, w, h = box
        cropped_region = cv2.resize(img[y:y+h, x:x+w], (45, 45))
        regions[i] = cropped_region

    return regions


if __name__ == "__main__": 
    print("\x1b[33m~[WARNING] This script is not meant to be executed.\x1b[0m"); 