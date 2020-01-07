#Okan Dönmez
#150170708

import cv2
import numpy as np
from moviepy.editor import *

def triangleTransformation(source, target):
    blank_image = np.zeros((500, 500, 3), np.uint8)
    blank_image[:,0:500] = (0,0,0)

    blank_image2 = np.zeros((500, 500, 3), np.uint8)
    blank_image2[:,0:500] = (0,0,0)

    oneHorizontal = (target[0][0] - source[0][0]) / 20
    oneVertical = (target[0][1] - source[0][1]) / 20

    twoHorizontal = (target[1][0] - source[1][0]) / 20
    twoVertical = (target[1][1] - source[1][1]) / 20

    threeHorizontal = (target[2][0] - source[2][0]) / 20
    threeVertical = (target[2][1] - source[2][1]) / 20


    blank_image2 = cv2.polylines(blank_image2, [source], True, (0,255,0),2)
    blank_image2 = cv2.polylines(blank_image2, [target], True, (0,255,0), 2)

    cv2.imshow("source and target", blank_image2)
    cv2.imwrite("outputs/part4_Source_and_Target.png", blank_image2)
    cv2.waitKey()

    images = []

    for i in range(21):
        triangle = np.array([[source[0][0] + (i * oneHorizontal), source[0][1] + (i * oneVertical)],
                    [source[1][0] + (i * twoHorizontal), source[1][1] + (i * twoVertical)],
                    [source[2][0] + (i * threeHorizontal), source[2][1] + (i * threeVertical)]], np.int32)

        blank_image[:, 0:500] = (0, 0, 0)
        cv2.polylines(blank_image, [triangle], True, ((0 + (i*12)), 0, (255 - (i*12))), 1)
        cv2.fillPoly(blank_image, [triangle], color=((0 + (i*12)), 0, (255 - (i*12))))

        temp = blank_image.copy()
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        images.append(temp)


    clip = ImageSequenceClip(images, fps=25)
    clip.write_videofile('outputs/part4_Result_Video' + '.mp4', codec="mpeg4")

if __name__ == "__main__":
    source = np.array([[50, 20], [170, 70], [150, 180]], np.int32)
    target = np.array([[330, 390], [400, 200], [450, 480]], np.int32)
    triangleTransformation(source, target)












