
"""
Image Preprocessing Techniques
Includes: Noise Reduction, Rotation Correction, Contrast Enhancement, Border Removal
"""

# choosen preprocessing 

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import cv2
import pytesseract
from pytesseract import Output
import imutils
from logger.logger_config import Logger


log = Logger.get_logger(__name__)

class ImagePreprocessor:
    """A class for various image preprocessing operations"""
    
    def __init__(self,image_array=None):
        """
        Initialize with image
        
        Args:
           
            image_array: Numpy array of image
        """
      
        if image_array is not None:
            self.image = image_array
        else:
            raise ValueError("image must be provided")
        
        self.original = self.image.copy()

    def reset(self):
        """Reset to original image"""
        self.image = self.original.copy()
        return self.image
    
    def get_image(self):
        """Get current processed image"""
        return self.image
    
    def save(self, output_path):
        """Save processed image"""
        cv2.imwrite(output_path, self.image)
        log.info(f"Image saved to {output_path}")

    
    # Noise Reduction Methods

    def gaussian_blur(self, kernel_size=(5, 5)):
        """
        Apply Gaussian blur for noise reduction
        Best for: General purpose noise reduction
        
        Args:
            kernel_size: Tuple of kernel dimensions (must be odd)
        """
        self.image = cv2.GaussianBlur(self.image, kernel_size, 0)
        return self.image
    
    def median_blur(self, kernel_size=5):
        """
        Apply median blur for noise reduction
        Best for: Salt-and-pepper noise
        
        Args:
            kernel_size: Size of kernel (must be odd)
        """
        self.image = cv2.medianBlur(self.image, kernel_size)
        return self.image
    
    def bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filter for noise reduction
        Best for: Preserving edges while reducing noise
        
        Args:
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
        """
        self.image = cv2.bilateralFilter(self.image, d, sigma_color, sigma_space)
        return self.image
    
    def non_local_means_denoising(self, h=10, template_window=7, search_window=21):
        """
        Apply Non-Local Means Denoising
        Best for: High-quality noise reduction (slower)
        
        Args:
            h: Filter strength
            template_window: Template patch size
            search_window: Search area size
        """
        if len(self.image.shape) == 3:
            self.image = cv2.fastNlMeansDenoisingColored(
                self.image, None, h, h, template_window, search_window
            )
        else:
            self.image = cv2.fastNlMeansDenoising(
                self.image, None, h, template_window, search_window
            )
        return self.image
    

    # Orientation Correction    
    def rotate_image(self):
       # Convert to RGB (Pytesseract often works better with RGB)
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Detect orientation and script
        results = pytesseract.image_to_osd(rgb, config='--psm 0 -c min_characters_to_try=5', output_type=Output.DICT)

        # Extract rotation angle
        rotate_angle = results["rotate"]

        # Rotate the image to correct the orientation
        self.image = imutils.rotate_bound(self.image, angle=rotate_angle)
        return self.image

    # Contrast Enhancement Methods
    
    def histogram_equalization(self):
        """
        Apply histogram equalization for contrast enhancement
        Best for: Improving overall contrast
        """
        if len(self.image.shape) == 3:
            # Convert to YUV color space
            yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
            # Equalize the Y channel
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            # Convert back to BGR
            self.image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            self.image = cv2.equalizeHist(self.image)
        
        return self.image
    
    def clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Best for: Local contrast enhancement without over-amplifying noise
        
        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        if len(self.image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            # Apply CLAHE to L channel
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            # Convert back to BGR
            self.image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            self.image = clahe.apply(self.image)
        
        return self.image
    
    def adjust_gamma(self, gamma=1.0):
        """
        Apply gamma correction for brightness adjustment
        
        Args:
            gamma: Gamma value (< 1 = brighter, > 1 = darker)
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        
        self.image = cv2.LUT(self.image, table)
        return self.image
    
    def adaptive_threshold(self, block_size=11, c=2):
        """
        Apply adaptive thresholding for binary conversion
        Best for: Document processing
        
        Args:
            block_size: Size of pixel neighborhood (must be odd)
            c: Constant subtracted from mean
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        self.image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, c)
        return self.image
    
    #  Border Removal Methods
    
    def remove_border_simple(self, border_size=10):
        """
        Remove border by cropping fixed pixels from edges
        
        Args:
            border_size: Number of pixels to remove from each edge
        """
        h, w = self.image.shape[:2]
        self.image = self.image[border_size:h-border_size, border_size:w-border_size]
        return self.image
    
    def remove_border_adaptive(self, threshold=240):
        """
        Remove white/light borders adaptively
        
        Args:
            threshold: Intensity threshold for detecting borders (0-255)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        
        # Find rows and columns that are mostly white
        row_means = np.mean(gray, axis=1)
        col_means = np.mean(gray, axis=0)
        
        # Find content boundaries
        rows_with_content = np.where(row_means < threshold)[0]
        cols_with_content = np.where(col_means < threshold)[0]
        
        if len(rows_with_content) > 0 and len(cols_with_content) > 0:
            top = rows_with_content[0]
            bottom = rows_with_content[-1] + 1
            left = cols_with_content[0]
            right = cols_with_content[-1] + 1
            
            self.image = self.image[top:bottom, left:right]
        
        return self.image
    
    def detect_and_crop_document(self):
        """
        Detect document boundaries and crop
        Best for: Document images with dark borders
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Crop image
            self.image = self.image[y:y+h, x:x+w]
        
        return self.image
    


if __name__ == "__main__":


    preprocessor = ImagePreprocessor("archive/dataset/Scientific/40024983-4986.jpg")
    preprocessor.rotate_image()
    preprocessor.bilateral_filter()
    preprocessor.clahe()
    preprocessor.remove_border_adaptive()
    preprocessor.get_image()
