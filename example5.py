import cv2
import numpy as np
from examples.example2 import binary_output

def perspective_transform_binary(binary_img):
    """Primenjuje perspektivnu transformaciju na binarnu sliku bez gubitka kvaliteta."""
    img_size = (binary_img.shape[1], binary_img.shape[0])

    # Definisanje koordinata na originalnoj slici
    src = np.float32([
        [585, 460],  # Leva gornja tačka
        [705, 460],  # Desna gornja tačka
        [1100, 720], # Desna donja tačka
        [200, 720]   # Leva donja tačka
    ])

    # Definisanje koordinata u "birds-eye view"
    dst = np.float32([
        [300, 0],     # Leva gornja tačka
        [1000, 0],    # Desna gornja tačka
        [1000, 720],  # Desna donja tačka
        [300, 720]    # Leva donja tačka
    ])

    # Izračunavanje matrice transformacije
    M = cv2.getPerspectiveTransform(src, dst)

    # Primena transformacije sa NEAREST interpolacijom
    warped = cv2.warpPerspective(binary_img, M, img_size, flags=cv2.INTER_NEAREST)

    return warped, M

def bird_eye(binary_image_frame):
    # Učitavanje binarne slike
    try:
        binary_img = cv2.imread(binary_image_frame, cv2.IMREAD_GRAYSCALE)
    except:
        binary_img = binary_image_frame

    # Provera da li su vrednosti binarne (0 i 255)
    if np.unique(binary_img).tolist() not in [[0, 255], [0]]:
        print("Upozorenje: Slika nije binarna!")

    # Primena transformacije
    warped_binary_img, perspective_matrix = perspective_transform_binary(binary_img)

    # Prikaz rezultata
    # cv2.imshow('Original Binary Image', binary_img)
    #cv2.imshow('Warped Binary Image (Birds-Eye View)', warped_binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Čuvanje transformisane slike
    cv2.imwrite('../output_images/warped_binary_image.jpg', warped_binary_img)

    return warped_binary_img

if __name__ == "__main__":
    bird_eye('../output_images/binary_output.jpg')
