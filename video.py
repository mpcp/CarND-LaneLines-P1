
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

#this is the solution for ffmpeg error
import imageio
imageio.plugins.ffmpeg.download()

from moviepy.editor import VideoFileClip
from IPython.display import HTML

from lane_func import *

def process_image(image):
    # image = mpimg.imread(image)
    # plt.imshow(image)
    # plt.show()

    kernel_size = 5
    low_threshold = 50
    high_threshold = 180

    # convert image to gray
    gray = grayimage(image)
    blur_gray = gaussianblur(gray, kernel_size)
    edges = cannygray(blur_gray, low_threshold, high_threshold)

    # display
    # plt.imshow(edges,cmap='Greys_r')
    # plt.show()

    # masked edges
    # mask_edges = masked_edges(image,edges)
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (440, 330), (540, 330), (imshape[1], imshape[0])]], dtype=np.int32)
    # (440,320),(530,320)
    # (450,290,490,290)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    mask_edges = cv2.bitwise_and(edges, mask)

    # houghline
    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_length = 40
    max_line_gap = 20
    combo = hough_lines(mask_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # combat = cv2.add()
    # combat = cv2.addWeighted(combo,0.8,image,1,0);
    combat = cv2.add(combo, image)
    #plt.imshow(combat)
    #plt.show()
    return combat
    #mpimg.imsave('test_images_out/' + pic, combo)


white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip('test_videos/solidWhiteRight.mp4')
white_clip= clip1.fl_image(process_image)
white_clip.write_videofile(white_output,audio=False)

