import cv2
import numpy as np

def process_image(img):
    ''' 
    The NVIDIA Paper suggests using a YUV color space this is nice because brightness is really easy in this color space
    '''
    img = img[70:img.shape[0]-26, :, :]
    img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def adjust_brigtness(img, amount):
    '''
    Adjusts the brightness of a YUV image, this only requires adjusting the Y component, however, care must be taken as an overflow can occur.
    '''
    Y = img[:,:,0]
    if(amount < 0):
        mask = Y > -amount
    else:
        mask = Y < 255 - amount
    Y = Y.astype(int, copy = False)
    Y += mask.astype(int)*amount                # False = 0, True = 1
    Y = Y.astype(np.uint8, copy = False)
    img2 = img.copy()
    img2[:,:,0] = Y
    return img2
