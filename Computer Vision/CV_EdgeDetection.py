import cv2
import numpy as np

original_image = cv2.imread('bird.jpg', cv2.IMREAD_COLOR)

# in this, we have to transform this colored image to grayscale

# opencv handles BGR instead of RGB
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Laplacian Kernel
# kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
# print(kernel)

# edged_image = cv2.filter2D(original_image, -1, kernel)

edged_image = cv2.Laplacian(gray_image, -1)


cv2.imshow("Original Image", original_image)
cv2.imshow("GrayScaled Image", gray_image)
cv2.imshow("Edged Image", edged_image)

cv2.waitKey(0)
cv2.destroyAllWindows()