from __future__ import division
__author__ = 'Naleen'

import cv2
import numpy as np

__author__ = 'Naleen'
import cv2
import numpy as np
    # char_top=character[0:20 ,xMin:xMax]
    # char_top_left=character[0:20 ,xMin:xMin+(xMax-xMin)/2]
    # char_top_right=character[0:20 ,xMin+(xMax-xMin)/2:xMax]
    # char_middle=character[20:44, xMin:xMax]
    # char_bottom=character[44:64, xMin:xMax]
    # char_bottom_left=character[44:64 ,xMin:xMin+(xMax-xMin)/2]
    # char_bottom_right=character[44:64 ,xMin+(xMax-xMin)/2:xMax]
#
# def zone_info_lines(char, character, y1, y2,x1, x2):
#
#     edges = cv2.Canny(character,2,3,apertureSize = 3)
#     minLineLength = 10
#     maxLineGap = 1
#     lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#     for x1,y1,x2,y2 in lines[0]:
#          cv2.line(char,(x1,y1),(x2,y2),(0,255,0),2)
#     print lines
#     cv2.imshow("char", char)
#     cv2.waitKey(0)
#     return "np.reshape(char_trimmed_array, 16*16).tolist()"


def zone_info_matrix(character, y1, y2,x1, x2):

    char_trimmed=character[y1:y2 ,x1:x2]

    character_resized = cv2.resize(char_trimmed,None,fx=16/(x2-x1), fy=16/(y2-y1), interpolation = cv2.INTER_CUBIC)


    (thresh, char) = cv2.threshold(character_resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    char = cv2.threshold(character_resized, thresh, 255, cv2.THRESH_BINARY)[1]

    #print "xMax:"+str(x2)+"   xMin:"+str(x1)
    char_trimmed_array=np.divide(np.array(char), 255)
    #print np.reshape(char_trimmed_array, 64*48)
    # cv2.imshow("char", char)
    # cv2.waitKey(0)

    return np.reshape(char_trimmed_array, 16*16).tolist()





def zone_info_horizontal(character, y1, y2,x1, x2):

    char_trimmed=character[y1:y2 ,x1:x2]
    char_trimmed_array=np.divide(np.array(char_trimmed).sum(axis=1), 255)

    return char_trimmed_array.tolist()
#
def zone_info_vertical(character, y1, y2,x1, x2):

    char_trimmed=character[y1:y2 ,x1:x2]

    character_resized = cv2.resize(char_trimmed,None,fx=50/(x2-x1), fy=1, interpolation = cv2.INTER_CUBIC)


    (thresh, char) = cv2.threshold(character_resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    char = cv2.threshold(character_resized, thresh, 255, cv2.THRESH_BINARY)[1]

    #print "xMax:"+str(x2)+"   xMin:"+str(x1)
    char_trimmed_array=np.divide(np.array(char).sum(axis=0), 255)
    #print char_trimmed_array
    # cv2.imshow("char", char)
    # cv2.waitKey(0)

    return char_trimmed_array.tolist()










# image : input image, cv2.imread()
def char_info(character, xMin, xMax, yMin, yMax):

    character_copy = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(character_copy, 127, 255, 0)

    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    i=0
    total_arc_length=0
    for i in range(0, len(contours)):
        total_arc_length=total_arc_length+cv2.arcLength(contours[i], True)

    char_ratio=int((yMax-yMin)/(xMax-xMin))


    return int(total_arc_length), char_ratio
#



