import cv2
import numpy as np
import matplotlib.pyplot as plt

def warp_to_bird_eye(img, src_points, dst_points):
    h, w = img.shape[:2]
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bird_eye = cv2.warpPerspective(img, M, (w, h))
    return bird_eye

# Učitavanje slika
img1 = cv2.imread('../test_images/solidYellowCurve2.jpg')
img2 = cv2.imread('../test_images/test2.jpg')

# Pretvaranje u sivi format za rad sa binarnim slikama
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Izrada binarne slike
_, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
_, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)

# Definišemo tačke za perspektivnu transformaciju
src_points = np.float32([[580, 460], [700, 460], [1120, 720], [200, 720]])
dst_points = np.float32([[200, 0], [1080, 0], [1080, 720], [200, 720]])

# Transformacija u bird's eye perspektivu
bird_eye1 = warp_to_bird_eye(binary1, src_points, dst_points)
bird_eye2 = warp_to_bird_eye(binary2, src_points, dst_points)

# Prikaz originalnih i transformisanih slika
titles = ['Challange Original', 'Test Original', 'Challange Bird Eye', 'Test Bird Eye']
images = [binary1, binary2, bird_eye1, bird_eye2]

plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
