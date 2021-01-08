import cv2
import numpy as np
from func import *


# Reading img
img = cv2.imread('resources/test_1.png')
cv2.namedWindow('img')

# Processing image
img = img_preprocessing(img)
cv2.imwrite('resources/pre.png', img)
get_circles(img)

# Getting results
results = checking_right_answers(img)
print(results)


def main():
    while True:
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
