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

    # Convert to RGB to draw rectangles on top of the image in debug mode. 
    if DEBUG == 2: 
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Convert img to grayscale and threshold it for the contour detection. 
    _, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Dilate the image on the y axis to be able to detect 2 or more part symbol 
    # like - and /. 
    kernel = np.ones((IMG_HEIGHT * 2, 1), np.uint8) 
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)
    
    # Perform the contour detection. 
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        # Append every contour to the bounding_boxes list. 
        x, y, w, h = cv2.boundingRect(contour)

        if (w < IMG_WIDTH and h < IMG_HEIGHT): 
            continue

        bounding_boxes.append((x, y, w, h))

        # Add rectangle to show bounding box in debug mode. 
        if DEBUG == 2: 
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show each symbol on separate window in debug mode. 
    if DEBUG == 2: 
        cv2.imshow("Fusion via dilatation", output) 
        cv2.imshow("dilated img", dilated) 

    return bounding_boxes


def bounding_box_sorting(bounding_boxes: list) -> list:
    """
    

    :param bounding_boxes: _description_
    :return: _description_
    """
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
    regions = np.zeros((region_count, IMG_WIDTH, IMG_HEIGHT), dtype=np.uint8)

    # Retrieve each region from the images. 
    for i, box in enumerate(bounding_boxes): 
        x, y, w, h = box
        cropped_region = cv2.resize(img[y:y+h, x:x+w], (45, 45))
        regions[i] = cropped_region

        if DEBUG == 2:
            cv2.imshow(f"{i}", cropped_region)

    return regions




if __name__ == "__main__": 
    print("\x1b[33m~[WARNING] This script is not meant to be executed.\x1b[0m"); 