import cv2
import numpy as np

from example2 import binary_output
from example5 import bird_eye
from example6 import lane_detection
from example7 import original_image_lane_detection

img = '../test_images/test2.jpg'

# example2.binary_output(img)
binary_output(img)
bird_eye()
lane_detection()
original_image_lane_detection(img)