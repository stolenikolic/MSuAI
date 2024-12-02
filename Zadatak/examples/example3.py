import cv2
import numpy as np

def perspective_transform(img):
    img_size = (img.shape[1], img.shape[0])

    # Definisanje koordinata na originalnoj slici
    src = np.float32([
        [360, 460],  # Leva gornja tačka
        [850, 460],  # Desna gornja tačka
        [1500, 720],  # Desna donja tačka
        [200, 720]  # Leva donja tačka
    ])

    # Definisanje koordinata u "birds-eye view"
    dst = np.float32([
        [300, 0],  # Leva gornja tačka
        [1000, 0],  # Desna gornja tačka
        [1000, 720],  # Desna donja tačka
        [300, 720]  # Leva donja tačka
    ])
    # Izracunavanje matrice tranformacije
    M = cv2.getPerspectiveTransform(src, dst)

    # Primjenjivanje transofrmacije
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, M

# Ucitavanje slike
img = cv2.imread('../test_images/challange00111.jpg')

# Primjena transformacije
warped_img, perspective_matrix = perspective_transform(img)

# Prikaz rezultata
cv2.imshow('Original Image', img)
cv2.imshow('Warped Image (Birds-Eye View)', warped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Cuvanje transformisane slike
cv2.imwrite('../output_images/warped_image.jpg', warped_img)