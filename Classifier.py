from matplotlib.font_manager import pickle_dump

__author__ = 'Naleen'


import numpy as np
import cv2
import scripts.locate_character as chr

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

img = cv2.imread('C:/Users/Naleen/PycharmProjects/CharReco/data/charsSK.jpg')
#img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
iMax = 73
jMax = 55
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

##output results##
#output=[1,2,3,4,5,6,7,8,9,10,11,12,13,14, 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,6,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112]


#j55/i113
j=0
i=0
for j in range (1,jMax):
    for i in range (1, iMax):
        if (i==11 or i==12 or i==13  or i==14 or i==16 or i==18 or i==19 or i==20 or i==24 or i==25 or i==26 or i==28 or i==30 or i==31):
            i=i+1
        else:

            character = img[64*(j-1):64*j, 128*(i-1):128*i]

            # xMin, xMax, yMin, yMax=chr.char_location(character)
            # print "xmin:"+str(xMin) + "   " +"yMin" +str(yMin)
            # print "xmax:"+str(xMax) + "   " +"yMax" +str(yMax)


            x=np.array(character)
            y=np.argmin(x, 1)
            values=y.tolist()

            # cv2.imshow(str(i)+'image',character)
            # cv2.waitKey(0)

##########################################################
######test##############
file = open("data.tab", "w")
classAttributes="val\t"
dataAttributes=""
for k in range (0, len(values)):
    dataAttributes=dataAttributes+"data"+str(k)+"\t"
classType="d\t"
dataType=""

for k in range (0, len(values)):
    dataType=dataType+"c"+"\t"
file.write(classAttributes+dataAttributes+"\n"+classType+dataType+"\nclass\n")

for j in range (1,jMax):
    for i in range (1, iMax):
        if (i==11 or i==12 or i==13  or i==14 or i==16 or i==18 or i==19 or i==20 or i==24 or i==25 or i==26 or i==28 or i==30 or i==31):
            i=i+1
        else:
            ####tab file####
            character = gray[64*(j-1):64*j, 128*(i-1):128*i]
            character1 = img[64*(j-1):64*j, 128*(i-1):128*i]

            xMin, xMax, yMin, yMax=chr.char_location(character1)
            print "xmin:"+str(xMin) + "   " +"yMin" +str(yMin)
            print "xmax:"+str(xMax) + "   " +"yMax" +str(yMax)
            x=np.array(character)
            y=np.argmax(x, 1)
            z=y.tolist()

            #values=y.tolist()
            file.write(str(i)+"\t")
            file.write('\t'.join(map(str,z)))
            file.write('\n')
file.close()
###########################################




data_training = Orange.data.Table ('data')
print '*******'

classifier = orngSVM.SVMLearner(data_training)
# classifierANN = Orange.classification.neural.NeuralNetworkLearner(data_training, n_mid=10, reg_fact=1, max_iter=300, normalize=True, rand=None)
# pickle.dump(classifierANN, open('ANN', 'w'))

# later:

classifier = pickle.load(open('ANN'))
data_validation = Orange.data.Table('valid.tab')
for e in data_validation:
    #print e
    print classifier(e, Orange.classification.Classifier.GetBoth)
#Orange.classification.Classifier.__call__(data_validation, w=0)
# orange.Learner.__call__(data_validation[1], w=0)
# print 'predictions:'
# for e in data_validation:
#       print orange.Learner.__call__(data_validation[e], w=0)