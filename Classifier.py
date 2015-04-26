from matplotlib.font_manager import pickle_dump

__author__ = 'Naleen'


import numpy as np
import cv2
import scripts.locate_character as chr
import scripts.zone_info as chr1
import scripts.map_char as MapChar


from matplotlib import pyplot as plt
from pybrain.datasets import SupervisedDataSet
import orange as orange
import Orange
import orngSVM
import pickle



from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.structure import SigmoidLayer



np.set_printoptions(threshold='nan')

img = cv2.imread('C:/Users/Naleen/PycharmProjects/CharReco/data/skel.png')
#img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
iMax = 73
jMax = 55
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#output=[1,2,3,4,5,6,7,8,9,10,11,12,13,14, 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,6,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112]
#j55/i113
j=0
i=0

##########################################################
######test##############

dataString=""
for j in range (1,jMax):
    for i in range (1, iMax):
        if (i==1 or i==2 or i==3 or i==4 or i==5 or i==6 or i==7 or i==8 or i==9 or i==10 or i==11 or i==12 or i==13  or i==14 or i==16 or i==18 or i==19 or i==20 or i==24 or i==25 or i==26 or i==28 or i==30 or i==31):
            i=i+1
        else:
            ####tab file####
            character = gray[64*(j-1):64*j, 128*(i-1):128*i]
            character1 = img[64*(j-1):64*j, 128*(i-1):128*i]

            xMin, xMax, yMin, yMax=chr.char_location(character1)
            total_arc_length, char_ratio=chr1.char_info(character1, xMin, xMax, yMin, yMax)

            # print "xmin:"+str(xMin) + "   " +"yMin" +str(yMin)
            # print "xmax:"+str(xMax) + "   " +"yMax" +str(yMax)



            #
            y=np.argmax(np.array(character), 1)
            dataArray=[]

            #dataArray=y.tolist()
            dataArray.append(xMin)
            dataArray.append(xMax)
            dataArray.append(yMin)
            dataArray.append(yMax)
            dataArray.append(total_arc_length)
            dataArray.append(char_ratio)
            dataArray.append(xMax-xMin)  #?
            dataArray.append(yMax-yMin)  #?

            char_hor_top = chr1.zone_info_horizontal(character, 0, 20, xMin, xMax)
            char_ver_top = chr1.zone_info_vertical(character, 0, 20, xMin, xMax)
            dataArray.extend(char_hor_top)
            dataArray.extend(char_ver_top)

            char_hor_top_left=chr1.zone_info_horizontal(character, 0, 20, xMin, xMin+(xMax-xMin)/2)
            dataArray.extend(char_hor_top_left)

            char_hor_top_right=chr1.zone_info_horizontal(character, 0, 20, xMin+(xMax-xMin)/2, xMax)
            dataArray.extend(char_hor_top_right)

            char_hor_bottom=chr1.zone_info_horizontal(character, 44, 64, xMin, xMax)
            char_ver_bottom=chr1.zone_info_vertical(character, 44, 64, xMin, xMax)
            dataArray.extend(char_hor_bottom)
            dataArray.extend(char_ver_bottom)

            char_hor_bottom_left=chr1.zone_info_horizontal(character, 44, 64, xMin, xMin+(xMax-xMin)/2)
            dataArray.extend(char_hor_bottom_left)

            char_hor_bottom_right=chr1.zone_info_horizontal(character,44, 64, xMin+(xMax-xMin)/2, xMax)
            dataArray.extend(char_hor_bottom_right)

            char_hor_middle=chr1.zone_info_horizontal(character,20, 44, xMin, xMax)
            char_ver_middle=chr1.zone_info_vertical(character,20, 44, xMin, xMax)
            dataArray.extend(char_hor_middle)
            dataArray.extend(char_ver_middle)

            char_pixel_matrix=chr1.zone_info_matrix(character,yMin, yMax, xMin, xMax)
            dataArray.extend(char_pixel_matrix)







            dataString=dataString+(str(i)+"\t")+('\t'.join(map(str,dataArray)))+('\n')

            #dataString=dataString+(str(i)+"\t")+('\n') #targets
            #dataString=dataString+("")+('\t'.join(map(str,dataArray)))+('\n') #insputs




file = open("data.tab", "w")
classAttributes="val\t"
dataAttributes=""
for k in range (0, len(dataArray)):
    dataAttributes=dataAttributes+"data"+str(k)+"\t"

classType="d\t"
dataType=""
for k in range (0, len(dataArray)):
    dataType=dataType+"c"+"\t"

file.write(classAttributes+dataAttributes+"\n"+classType+dataType+"\nclass\n")
file.write(dataString.encode("UTF-8"))
file.close()
# print dataString
# ###########################################
#
#
#
# #
#data_training = Orange.data.Table ('data')
# print '*******'
# print len(dataArray)
# #
# #
#classifier = orngSVM.SVMLearner(data_training)
#classifierANN = Orange.classification.neural.NeuralNetworkLearner(data_training, n_mid=10, reg_fact=1, max_iter=300, normalize=True, rand=None)
#pickle.dump(classifierANN, open('ANN', 'w'))
#
# # later:
#
# classifier = pickle.load(open('ANN'))
# data_validation = Orange.data.Table('valid.tab')
# for e in data_validation:
# #     #print e
#      print classifier(e, Orange.classification.Classifier.GetBoth)
# #Orange.classification.Classifier.__call__(data_validation, w=0)
# # orange.Learner.__call__(data_validation[1], w=0)
# # print 'predictions:'
# # for e in data_validation:
# #       print orange.Learner.__call__(data_validation[e], w=0)