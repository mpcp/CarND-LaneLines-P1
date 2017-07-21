import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

from pipeline_func import *

pics = os.listdir('test_images')
for index, pic in enumerate(pics):
    plt.figure(index)

    # Read in and grayscale the image
    image = mpimg.imread('test_images/' + pic)  # do not use cv2.imread()
    gray = grayscale(image)
    # plt.imshow(gray)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    # plt.imshow(blur_gray)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 200
    edges = canny(blur_gray, low_threshold, high_threshold)
    # plt.imshow(edges)

    # Next we'll create a masked edges image using cv2.fillPoly()
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (440, 320), (530, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    # plt.imshow(masked_edges)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    # plt.imshow(line_image)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))
    # plt.imshow(color_edges)

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(image, 1, line_image, 1, 0)
    # lines_edges = cv2.addWeighted(color_edges, 1, line_image, 1, 0)

    plt.imshow(lines_edges)
    plt.show()

    mpimg.imsave('test_images_output/' + pic, lines_edges)
