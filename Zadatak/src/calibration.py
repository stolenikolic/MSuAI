import cv2
import numpy as np
import glob

# Priprema objektnih tacaka
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Liste za objektne i slikovne tacke
objpoints = []  # 3D tacke iz stvarnog svijeta
imgpoints = []  # 2D tacke sa slika

# Ucitavanje slika sahovske table
images = glob.glob('../camera_cal/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pronalazenje uglova sahovske table
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Crtanje uglova na slici
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Kalibracija kamere
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Ispravljanje distorzije na slici
img = cv2.imread('../camera_cal/calibration2.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)

# Cuvanje output slike
cv2.imwrite('../output_images/undistorted.jpg', dst)