import cv2
import numpy as np


def region_of_interest(image, region_points):

    # we are going to replace pixels with 0 (black) - the regions we're not interested in
    mask = np.zeros_like(image)

    # the region that we're interested in is the lower triangle - 255 white pixels
    cv2.fillPoly(mask, region_points, 255)

    # we have to use the mask: we want to keep the regions of the original image where
    # the mask has white colored pixels
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def get_detected_lanes(image):
    (height, width) = (image.shape[0], image.shape[1])
    # print(height)
    # print(width)

    # we have to turn the image into grayscale

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edge detection algorithm -> canny's algorithm -> optimal detector
    # here lowerthreshold = 100 and upperthreshold = 120
    canny_image = cv2.Canny(gray_image, 100, 120)

    # we're interested in the "lower region" of the image (driving lanes only)

    region_of_interested_vertices = [
        (0, height),
        (width/2, height*0.65),
        (width, height)
    ]

    # we can get of the un relevant part of the image, just keep the lower triangle region

    cropped_image = region_of_interest(canny_image, np.array([region_of_interested_vertices], np.int32))

    return cropped_image

# video = several frames (images shown after each other)
video = cv2.VideoCapture('lane_detection_video.mp4')

while video.isOpened():

    is_grabbed, frame = video.read()

    # end of the video

    if not is_grabbed:
        break


    frame = get_detected_lanes(frame)


    cv2.imshow("Lane Detection Video", frame)
    cv2.waitKey(20)


video.release()
cv2.destroyAllWindows()