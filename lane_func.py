#Find Lane Line
#This is common functions

#import headers
import cv2
import numpy as np

#convert image to grayscale
def grayimage(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

#gaussian smoothing
def gaussianblur(gray,kernel_size):
    return cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

#canny edge detection
def cannygray(gray,low_threshold,high_threshold):
    return cv2.Canny(gray,low_threshold,high_threshold)

#define masked edges
def masked_edges(image,edges):
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450,290),(450,290),(imshape[1],imshape[0])]],dtype=np.int32)
    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges,mask)
    return masked_edges


#hough transformation to lines
def hough_lines(image,rho,theta,threshold,min_line_length,max_line_gap):
    line_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    #hough lines
    lines = cv2.HoughLinesP(image,rho,theta,threshold,np.array([]),min_line_length,max_line_gap)

    #draw lines on the blank
    #left lines parameters
    xlmin = image.shape[1]
    ylmin = image.shape[0]
    xlmax = 0
    ylmax = 0

    #right lines parameters
    xrmin = image.shape[1]
    yrmin = image.shape[0]
    xrmax = 0
    yrmax = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            #cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
            threshold = (y2-y1)/(x2-x1)
            if(threshold>0):            #left
                xlmin = min(min(x1,x2),xlmin)
                ylmin = min(min(y1,y2),ylmin)
                xlmax = max(max(x1,x2),xlmax)
                ylmax = max(max(y1,y2),ylmax)
            else:
                xrmin = min(min(x1,x2),xrmin)
                yrmin = min(min(y1,y2),yrmin)
                xrmax = max(max(x1,x2),xrmax)
                yrmax = max(max(y1,y2),yrmax)

    cv2.line(line_image,(xlmin,ylmin),(xlmax,ylmax),(255,0,0),10)
    cv2.line(line_image,(xrmax,yrmin),(xrmin,yrmax),(255,0,0),10)


        #create a color binary image to combine with line image
    color_edges = np.dstack((image,image,image))

    combo = cv2.addWeighted(color_edges,0.8,line_image,1,0)
    return combo