
#import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

import math

#read an image
image = mpimg.imread("test_images/solidWhiteRight.jpg")
print("This image is :",type(image),"with dimensions:",image.shape)
plt.imshow(image,cmap="gray")
plt.show()
os.listdir("test_images/")

def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def canny(img,low_threshold,high_threshold):
    return cv2.Canny(img,low_threshold,high_threshold)

def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def region_of_interest(img,vertices):
    mask = np.zeros_like(img)

    if len(img.shape)>2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask,vertices,ignore_mask_color)

    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def draw_lines(img,lines,color=[255,0,0],thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)

def hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):

    lines = cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),minLineLength=min_line_len,maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)
    draw_lines(line_img,lines)

def weighted_img(img,initial_img,a=0.8,b=1.,c=0.):
    return cv2.addWeighted(initial_img,a,img,b,c)

image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
gray = grayscale(image)

kernel_size = 5
blur_gray = gaussian_blur(gray,kernel_size)

low_threshold = 50
high_threshold = 150
edges = canny(blur_gray,low_threshold,high_threshold)

imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450,290),(490,290),(imshape[1],imshape[0])]],dtype=np.int32)
masked_edges = region_of_interest(edges,vertices)

#define hough transformation
rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap = 20
line_image = np.copy(image)*0

xxx = hough_lines(masked_edges,rho,theta,threshold,min_line_length,max_line_gap)
color_edges = np.dstack((edges,edges,edges))

lines_edges = cv2.addWeighted(color_edges,0.8,line_image,1,0)
plt.imshow(lines_edges)
plt.show()


#videos
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
