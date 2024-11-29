import cv2
import numpy as np

def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

def color_threshold(img, thresh=(170, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def binary_output(frame):
    # Ucitavanje ispravljene slike
    try:
        img = cv2.imread(frame)
    except:
        img = frame

    # Primjena Sobel filtera
    grad_binary = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)

    # Primjena color filtera
    color_binary = color_threshold(img, thresh=(150, 255))

    # Kombinovanje rezultata
    combined_binary = np.zeros_like(grad_binary)
    combined_binary[(grad_binary == 1) | (color_binary == 1)] = 1

    # Cuvanje rezultata
    # cv2.imshow('Gradient Binary', grad_binary * 255)
    # cv2.imshow('Color Binary', color_binary * 255)
    #cv2.imshow('Combined Binary', combined_binary * 255)
    cv2.imwrite('../output_images/binary_output.jpg', combined_binary * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return combined_binary * 255

#binary_output('../test_images/test2.jpg')

if __name__ == "__main__":
    binary_image = binary_output('../test_images/test2.jpg')
    cv2.imshow('Combined Binary', binary_image)
    cv2.waitKey(1)