import cv2
import numpy as np


original_image = cv2.imread('bird.jpg', cv2.IMREAD_COLOR)

kernel = np.ones((5, 5)) / 25

print(kernel)

blur_image = cv2.filter2D(original_image, -1, kernel)
# -1 is destination depth => depth of original_image is equal to depth of output blur_image

cv2.imshow('Original Image', original_image)
cv2.imshow("Blurred Image", blur_image)

# gaussian blur is reduced to reduce noise

cv2.waitKey(0)
cv2.destroyAllWindows()