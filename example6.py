import cv2
import numpy as np

def detect_lane_pixels_and_fit_poly(binary_warped):
    """
    Detektuje piksele traka i fituje polinome za leve i desne granice traka.
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50
    window_height = binary_warped.shape[0] // nwindows
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def draw_lane_lines_on_original(binary_warped, left_fit, right_fit):
    """
    Crta pune linije za granice traka na originalnoj binarnoj slici bez izmene originalnih piksela.
    """
    # Kreiranje kopije slike u formatu BGR za crtanje
    out_img = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), dtype=np.uint8)
    out_img[binary_warped == 255] = [255, 255, 255]  # Prenesi bele piksele u RGB kopiju

    # Generisanje tačaka za polinome
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Iteracija kroz sve y vrednosti i crtanje linija
    for i in range(len(ploty) - 1):
        # Leva linija
        cv2.line(
            out_img,
            (int(left_fitx[i]), int(ploty[i])),
            (int(left_fitx[i + 1]), int(ploty[i + 1])),
            (255, 0, 0),  # Plava boja
            thickness=5
        )

        # Desna linija
        cv2.line(
            out_img,
            (int(right_fitx[i]), int(ploty[i])),
            (int(right_fitx[i + 1]), int(ploty[i + 1])),
            (0, 0, 255),  # Crvena boja
            thickness=5
        )

    return out_img

def lane_detection(binary_warped_frame):
    # Učitavanje binarne slike
    try:
        binary_warped = cv2.imread(binary_warped_frame, cv2.IMREAD_GRAYSCALE)
    except:
        binary_warped = binary_warped_frame

    # Detekcija piksela traka i fitovanje polinoma
    left_fit, right_fit = detect_lane_pixels_and_fit_poly(binary_warped)

    # Crtanje linija na originalnoj slici
    lane_lines_img = draw_lane_lines_on_original(binary_warped, left_fit, right_fit)

    # Prikaz slike
    # cv2.imshow("Lane Lines", lane_lines_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Čuvanje slike sa nacrtanim linijama
    cv2.imwrite('../output_images/lane_lines_image.jpg', lane_lines_img)

    return lane_lines_img

if __name__ == "__main__":
    lane_lines_img = lane_detection('../output_images/warped_binary_image.jpg')
    # cv2.imshow("Lane Lines", lane_lines_img)
    # cv2.waitKey(0)

