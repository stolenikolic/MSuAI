import cv2
import numpy as np
from lane_detection import detect_lane_pixels_and_fit_poly

def warp_lane_to_original(binary_warped, original_img, left_fit, right_fit, Minv):
    """
    Prebacuje detektovane linije iz bird's-eye perspektive na originalnu sliku.
    """
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    lane_img = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(lane_img, np.int_([pts]), (0, 255, 0))

    # Crtanje crvene leve linije
    for i in range(len(ploty) - 1):
        cv2.line(
            lane_img,
            (int(left_fitx[i]), int(ploty[i])),
            (int(left_fitx[i + 1]), int(ploty[i + 1])),
            (255, 0, 0),  # Crvena boja
            thickness=50
        )

    # Crtanje plave desne linije
    for i in range(len(ploty) - 1):
        cv2.line(
            lane_img,
            (int(right_fitx[i]), int(ploty[i])),
            (int(right_fitx[i + 1]), int(ploty[i + 1])),
            (0, 0, 255),  # Plava boja
            thickness=50
        )

    lane_on_original = cv2.warpPerspective(lane_img, Minv, (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, lane_on_original, 0.3, 0)

    return result

def original_image_lane_detection(frame, lanes_img):
    # Učitavanje binarne slike i originalne slike
    try:
        binary_warped = cv2.imread(lanes_img, cv2.IMREAD_GRAYSCALE)
        original_img = cv2.imread(frame)
    except:
        binary_warped = lanes_img
        original_img = frame

    # Definisanje perspektivnih tačaka
    src = np.float32([[585, 460], [705, 460], [1100, 720], [200, 720]])
    dst = np.float32([[300, 0], [1000, 0], [1000, 720], [300, 720]])

    # Matrica transformacije i invertovane transformacije
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Detekcija linija
    left_fit, right_fit = detect_lane_pixels_and_fit_poly(binary_warped)

    # Prebacivanje linija na originalnu sliku
    final_result = warp_lane_to_original(binary_warped, original_img, left_fit, right_fit, Minv)

    # Prikaz slike sa detektovanim trakama
    cv2.imshow("Final Result", final_result)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Čuvanje rezultata
    cv2.imwrite('../output_images/final_lane_result.jpg', final_result)

    return final_result

if __name__ == "__main__":
    original_image_lane_detection('../test_images/test6.jpg', '../output_images/lane_lines_image.jpg')