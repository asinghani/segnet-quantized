import numpy as np
import os, sys
import random
import cv2

from segnet import SegNet

import tensorflow as tf
from tensorflow import keras as K

from vis import view_seg_map

model = SegNet()
model.load_weights("model-0118.h5")

cap = cv2.VideoCapture(0)

while True:
    frame = cap.read()[1]
    frame = cv2.resize(frame[0:720, 280:1000], (128, 128))

    frame = (frame / 127.5) - 1.0

    p = model.predict(np.array([frame]))[0]
    p = p.reshape((128, 128, 2)).argmax(axis=2)
    image1 = ((frame + 1.0) * 127.5).astype(np.uint8)
    seg, overlay = view_seg_map(image1, p, color=(0, 255, 0), include_overlay=True)

    cv2.imshow("frame", cv2.resize(seg, (512, 512)))
    cv2.waitKey(5)
