# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:21:47 2017

@author: iwatts
"""
import numpy as np
import pickle
import time
import cv2
from keras.models import Model
from keras.models import load_model
from skimage.segmentation import felzenszwalb as felz


def normalize(im):
    return (np.float64(im) - 127)/127


t = time.time()

model = load_model('model.h5')
model_logit = Model(inputs=model.input,outputs=model.get_layer(index=len(model.layers)-1).output)
feat = normalize(pickle.load(open("../testFeat.pkl","rb")))
label = pickle.load(open("../testLabel.pkl","rb"))
samples = len(label)

pred = modelLogit.predict(feat)

pred_reshape = np.zeros(samples)
incorrect = np.zeros(samples)
for i in range(samples):
    predReshape[i] = np.where(pred[i,:] == np.max(pred[i,:]))[0][0]
    if predReshape[i] != label[i]:
        incorrect[i] = 1
        print("Image", i, "was predicted to be", int(predReshape[i]),
              "but was actually", str(label[i]) + '.')
    elif np.max(pred[i,:]) < 5:
        incorrect[i] = 1
        print("Image", i, "was predicted correctly but with too low confidence.")
classificationRate = (samples - sum(incorrect)) / samples
print("Correct classification rate was", classificationRate)

cap = cv2.VideoCapture('../drive.mp4')
ret, frame = cap.read()
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'XVID'),
                      5, (frame.shape[1], frame.shape[0]))

while ret:
    ret, frame = cap.read()
    if ret == False:
        break
    seg = felz(frame,scale=100,min_size=500)
    for i in range(1,int(np.max(seg)+1)):
        points = np.argwhere(seg == i)
        x = []
        y = []
        for point in points:
            x.append(point[1])
            y.append(point[0])
        UL = [min(x), min(y)]
        LR = [max(x), max(y)]
        w = LR[0] - UL[0]
        h = LR[1] - UL[1]
        LC = np.array([int((min(x)+max(x))/2), max(y)],np.float32)
        if w < 125 and w > 25 and h < 125 and h > 25:
            vidPred = modelLogit.predict(normalize(np.array([cv2.resize(frame[UL[1]:LR[1], UL[0]:LR[0]], (32,32))])))
            if np.max(vidPred) > 22:
                cv2.rectangle(frame, (UL[0], UL[1]), (LR[0], LR[1]), (255, 0, 0), 2)
                text = str(np.where(vidPred == np.max(vidPred))[1][0])
                cv2.putText(frame, text, (LC[0], LC[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
    out.write(frame)
out.release()

t2 = time.time()
print('Run time (s):', round(t2-t, 2))
