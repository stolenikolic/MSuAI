import cv2
import numpy as np

def detect_lane_pixels_and_fit_poly(binary_warped):
    """Детектује пикселе траке и уклапа полином."""
    # Израчунавање хистограма
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Хиперпараметри за sliding window
    nwindows = 9
    margin = 100
    minpix = 50

    window_height = int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    # Sliding window
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Детекција пиксела унутар прозора
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

    # Уклапање полинома
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_lane_inds, right_lane_inds

def visualize_lane_detection(binary_warped, left_fit, right_fit):
    """Приказује детекцију траке и уклопљене линије."""
    # Креирај празну слику у боји за визуализацију
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Креирај координате за уклопљене линије
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Цртање линија на слици
    for i in range(len(ploty) - 1):
        # Лева линија
        cv2.line(out_img,
                 (int(left_fitx[i]), int(ploty[i])),
                 (int(left_fitx[i + 1]), int(ploty[i + 1])),
                 (255, 0, 0), 10)
        # Десна линија
        cv2.line(out_img,
                 (int(right_fitx[i]), int(ploty[i])),
                 (int(right_fitx[i + 1]), int(ploty[i + 1])),
                 (0, 0, 255), 10)

    # Приказ резултата
    cv2.imshow("Lane Detection Visualization", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Чување слике
    cv2.imwrite('../output_images/lane_lines_image.jpg', out_img)

# Учитавање трансформисане бинарне слике
binary_warped = cv2.imread('../output_images/warped_binary_image.jpg', cv2.IMREAD_GRAYSCALE)
# binary_warped = cv2.imread('../test_images/straight_lines1.jpg', cv2.IMREAD_GRAYSCALE)

# Примена функција
left_fit, right_fit, _, _ = detect_lane_pixels_and_fit_poly(binary_warped)
visualize_lane_detection(binary_warped, left_fit, right_fit)


def warp_back_to_original(binary_warped, Minv, left_fit, right_fit, undistorted):
    """
    Враћа слику из птичје перспективе у оригиналну перспективу са визуелизацијом линија.

    binary_warped: бинарна слика у птичјој перспективи.
    Minv: инверзна матрица трансформације за враћање у оригиналну перспективу.
    left_fit: полином који представља леву линију.
    right_fit: полином који представља десну линију.
    undistorted: оригинална слика (исправљена за дисторзију).
    """
    # Генериши x и y вредности за уклопљене линије
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Створи празну слику за визуализацију
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    lane_area = np.dstack((warp_zero, warp_zero, warp_zero))

    # Попуни поље између линија зеленом бојом
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(lane_area, np.int_([pts]), (0, 255, 0))  # Зелена боја за траку

    # Нацртај леву ивицу (плава)
    for i in range(len(ploty) - 1):
        cv2.line(lane_area,
                 (int(left_fitx[i]), int(ploty[i])),
                 (int(left_fitx[i + 1]), int(ploty[i + 1])),
                 (255, 0, 0), 50)

    # Нацртај десну ивицу (црвена)
    for i in range(len(ploty) - 1):
        cv2.line(lane_area,
                 (int(right_fitx[i]), int(ploty[i])),
                 (int(right_fitx[i + 1]), int(ploty[i + 1])),
                 (0, 0, 255), 50)

    # Примени обрнуту перспективну трансформацију
    lane_area_warped = cv2.warpPerspective(lane_area, Minv, (binary_warped.shape[1], binary_warped.shape[0]))

    # Комбинуј са оригиналном сликом
    result = cv2.addWeighted(undistorted, 1, lane_area_warped, 0.3, 0)

    return result

# Дефиниши тачке
src = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])

# Матрице трансформације
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

undistorted = cv2.imread('../test_images/test3.jpg')

# Пример позива
final_result = warp_back_to_original(binary_warped, Minv, left_fit, right_fit, undistorted)
cv2.imwrite('../output_images/final_result.jpg', final_result)
cv2.imshow("Final Result", final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()