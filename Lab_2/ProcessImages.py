import os
import cv2
import glob
import imageio
import numpy as np
import skimage as ski
from utils import non_max_suppression, get_hsv_color_ranges

def load_images(input_files_dir: str, mode:str = "", type:str = "jpg") -> list:
    """
    Load images from a directory on different modes.
    
    Args
    ----
    - input_files_dir: path to the directory containing the images.
    - mode: mode in which the images should be loaded(rgb/bgr/gray).
    - type: type of the images to be loaded(jpg/png/both).
    
    Returns
    ----
    - images: list with the loaded images as mtrixes.
    """
    
    if type.lower() == "jpg": # Extract the paths of the images
        filenames = [path for path in glob.glob(f"{input_files_dir}/*jpg")] 
        
    elif type.lower() == "png":
        filenames = [path for path in glob.glob(f"{input_files_dir}/*png")]
        
    else: # Both types
        filenames = [path for path in glob.glob(f"{input_files_dir}/*jpg")] + [path for path in glob.glob(f"{input_files_dir}/*png")]
    
    if mode.lower() == "bgr":
        return [imageio.v2.imread(filename) for filename in filenames]
    
    elif mode.lower() == "gray":
        return [cv2.imread(filename, cv2.IMREAD_GRAYSCALE) for filename in filenames]
    
    else:
        return [cv2.imread(filename) for filename in filenames]
    
    
def show_images(imgs: list) -> None:
    """
    Display a set of images.
    
    Args
    ----
    - images: list with the images to be displayed.
    """
    
    for img in imgs:
        cv2.imshow("image", img.astype(np.uint8))
        cv2.waitKey(0) # Screen time
    
    cv2.destroyAllWindows()
    
def write_images(imgs: list, output_dir: str,  save_name: str) -> None:
    """
    Write a set of images to a directory.
    
    Args
    ----
    - output_files_dir: path to the directory where the images should be saved.
    - images: list with the images to be saved.
    - save_name: name of the images to be saved.
    """
    
    os.makedirs(output_dir, exist_ok = True)
    
    for i, img in enumerate(imgs):
        cv2.imwrite(f"{output_dir}/{save_name}_{i}.jpg", img)
    
def mask_segmentation(img: np.array, lower_bound: np.array, upper_bound: np.array) -> np.array:
    """
    Segment an image using a color mask.
    
    Args
    ----
    - img: image to be segmented.
    - lower_bound: lower bound of the color mask.
    - upper_bound: upper bound of the color mask.
    
    Returns
    ----
    - mask: mask of the color of image.
    - segmented_img: segmented original image.
    """
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # For a higher contrast
    mask = cv2.inRange(hsv, lower_bound, upper_bound) # Create the mask
    segmented_img = cv2.bitwise_and(img, img, mask = mask) # Apply the mask
    
    return mask, segmented_img

def hsv_selection(img:np.ndarray, mode:str = "select") -> tuple:
    """
    Returns the HSV values of the mask, either by selecting them or by introducing them.
    
    Args
    ----
    - img: image to be segmented.
    - mode: mode in which the values should be selected(select/introduce).
    
    Returns
    ----
    - lower_bound: lower bound of the color mask.
    - upper_bound: upper bound of the color mask."""
    
    if mode.lower() == "select":
        
        return get_hsv_color_ranges(img)
    
    else: # User introduces the values of the mask
        wrong_values = True
        
        while wrong_values:
            
            hMin = int(input("Introduce the minimum value for the H channel: "))
            sMin = int(input("Introduce the minimum value for the S channel: "))
            vMin = int(input("Introduce the minimum value for the V channel: "))
            
            hMax = int(input("Introduce the maximum value for the H channel: "))
            sMax = int(input("Introduce the maximum value for the S channel: "))
            vMax = int(input("Introduce the maximum value for the V channel: "))
            
            lower_bound = np.array([hMin, sMin, vMin])
            upper_bound = np.array([hMax, sMax, vMax])
            
            bounds = np.array([lower_bound, upper_bound])
            
            values_verified = 6
            
            for bound in bounds:
                if bound < 0 or bound > 255:
                    print("The values should be between 0 and 255")
                
                else:
                    values_verified -= 1
                    
            if values_verified == 0:
                wrong_values = False
                
        return lower_bound, upper_bound

def segment_images(input_dir: str = "data", mode:str = "jpg", save:bool = False, output_dir: str = "processed_data", save_name: str = "segmented", show:bool = True) -> None:
    
    imgs = load_images(input_dir, mode) # Load the images as matrixes of pixeles
    
    user_hsv_selection = input("Do you want to select the HSV values of the mask? (select): ")
    
    lower_bound, upper_bound = hsv_selection(imgs[0], "select")