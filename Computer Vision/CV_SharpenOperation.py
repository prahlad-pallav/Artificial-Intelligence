import cv2
import numpy as np

original_image = cv2.imread('unsharp_bird.jpg', cv2.IMREAD_COLOR)


# sharpen kernel => Face Recognition, Blurry CCTV footage => increase the precision of the underlying model
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
print(kernel)

sharpen_image = cv2.filter2D(original_image, -1, kernel)

cv2.imshow("Original Image", original_image)
cv2.imshow("Sharpen Image", sharpen_image)

cv2.waitKey(0)
cv2.destroyAllWindows()