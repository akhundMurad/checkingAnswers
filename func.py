import cv2
import numpy as np
import easyocr
import config
import operator


reader = easyocr.Reader(['en'])
checked = []


def get_roi(img_):
    """Getting the region of the interest"""

    # Finding the contours on the image
    contours, hierarchy = cv2.findContours(img_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # There will be the largest area of the contour
    max_area = 0
    # There will be the largest contour
    biggest = None

    # Finding the largest area and contour
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01*peri, True)

            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest


def img_preprocessing(img_):
    """Preprocessing image"""

    # Changing color scheme to grayscale
    img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    # Canny algorithm (finding edges)
    img_canny = cv2.Canny(img_gray, 85, 255)

    # Finding ROI
    roi = get_roi(img_canny)
    roi = list(roi.ravel())

    return img_[roi[1]:roi[3], roi[0]:roi[4]]


def get_circles(img_):
    """Getting the marked answers"""

    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

    # Thresholding image (we need only two colors to be)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    blur = cv2.medianBlur(gray, 5)

    # Hough Circles Transform
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=8, param2=30, minRadius=9, maxRadius=25)
    circles = np.uint16(np.around(circles))

    # Drawing the circles
    for (x, y, r) in circles[0, :]:
        nn_x = int(x + (r * 0.7255))
        nn_y = int(y + (r * 0.7255))
        prob = 0

        for pixels in thresh[y:nn_y, x:nn_x]:
            for pixel in pixels:
                if pixel == 0:
                    prob += 1

        if prob >= 65:
            cv2.circle(img_, (x, y), r, (0, 255, 0), 3)
            checked.append([x, y, r])


def getting_checked_answer(img_):
    all_roi = {}
    all_roi_coords = []
    i = 1
    for c in checked:
        x = c[0]
        y = c[1]
        r = c[2]
        r = int(r/2)
        roi_coords = [[x-r-3, x+r+3], [y+r+3, y-r-3]]
        all_roi_coords.append(roi_coords)
        roi = img_[roi_coords[1][1]:roi_coords[1][0], roi_coords[0][0]:roi_coords[0][1]]
        all_roi.update({i: [roi, y]})
        i += 1

    sorted_roi_list = []
    result = {}

    for i in range(1, len(all_roi)+1):
        sorted_roi_list.append(all_roi[i])

    sorted_roi_list = sorted(sorted_roi_list, key=operator.itemgetter(1))

    k = 1
    for sorted_roi in sorted_roi_list:
        result.update({k: sorted_roi[0]})
        k += 1

    return result


def checking_right_answers(img_):
    roi = getting_checked_answer(img_)
    answers = []
    done = {}

    for i in range(1, len(checked)+1):
        roi[i] = cv2.resize(roi[i], (64, 64))
        _, roi[i] = cv2.threshold(roi[i], 70, 255, cv2.THRESH_BINARY)
        answer = reader.readtext(roi[i], detail=0)
        answers.append(answer[0])

    answers = check_char(answers)

    for i, j in zip(range(len(answers)), range(1, len(checked)+1)):
        if answers[i] == config.answers[j]:
            done.update({j: 'RIGHT'})
        else:
            done.update({j: 'WRONG'})

    return done


def check_char(char_list):
    new_char_list = []
    options = ['A', 'B', 'C', 'D', 'E']
    for element in char_list:
        new_element = ''
        if len(element) == 1:
            new_char_list.append(element)
            continue
        else:
            for char in element:
                if char in options:
                    continue
                else:
                    element = element.replace(char, "")

            new_char_list.append(element)

    return new_char_list
