import cv2
import numpy as np

# print(cv2.__version__)

# image = cv2.imread('camus.jpg', cv2.IMREAD_GRAYSCALE)

image = cv2.imread('bird.jpg', cv2.IMREAD_COLOR)

# print(image)

# Values closed to zero means darker and value closed to one means brighter

print(image.shape)
print(np.amax(image))
print(np.amin(image))

cv2.imshow('Computer Vision', image)

cv2.waitKey(0)

cv2.destroyAllWindows()