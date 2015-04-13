__author__ = 'Naleen'
import cv2
import numpy as np


# image : input image, cv2.imread()
def zone_info(character, xMin, xMax, yMin, yMax):

    character_copy = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(character_copy, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)



    #print len(contours)

    character_copy = cv2.drawContours(character, contours, -1, (0,255,0), 3)

    i=0
    total_arc_length=0
    for i in range(0, len(contours)):
        total_arc_length=total_arc_length+cv2.arcLength(contours[i], True)

    char_ratio=int((yMax-yMin)/(xMax-xMin))

    char_top=character[0:20 ,xMin:xMax]
    char_top_left=character[0:20 ,xMin:xMin+(xMax-xMin)/2]
    char_top_right=character[0:20 ,xMin+(xMax-xMin)/2:xMax]

    char_middle=character[20:44, xMin:xMax]

    char_bottom=character[44:64, xMin:xMax]
    char_bottom_left=character[44:64 ,xMin:xMin+(xMax-xMin)/2]
    char_bottom_right=character[44:64 ,xMin+(xMax-xMin)/2:xMax]

    cv2.imshow('dst_rt', char_top)
    cv2.waitKey(0)




    return int(total_arc_length), char_ratio
#
