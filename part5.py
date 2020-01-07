#Okan Dönmez
#150170708

import numpy as np
import cv2
import dlib
import part2
from moviepy.editor import *
import sys

def pointToIndex(point, pointList):
    for i in range(len(pointList)):
        if (int(point[0]) == int(pointList[i][0]) and int(point[1]) == int(pointList[i][1])):
            return i
        else:
            None

def markTheHumanFace(fileName):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("inputs/shape_predictor_68_face_landmarks.dat")

    image = cv2.imread(fileName)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectangles = detector(gray)
    points = predictor(gray, rectangles[0])

    colorGreen = (0, 255, 0)
    colorBlue = (255, 0, 0)
    #cv2.rectangle(image, rectangleCoords[0], rectangleCoords[1], colorBlue, 1)

    imageCoordinates = []

    height = image.shape[0]
    width = image.shape[1]

    imageCoordinates.append((0, 0))
    imageCoordinates.append((width / 2, 0))
    imageCoordinates.append((width - 1, 0))
    imageCoordinates.append((width - 1, height / 2))
    imageCoordinates.append((width - 1, height - 1))
    imageCoordinates.append((width / 2, height - 1))
    imageCoordinates.append((0, height - 1))
    imageCoordinates.append((0, height / 2))

    for i in range(68):
        imageCoordinates.append(pointToTuple(points.part(i)))
        cv2.circle(image, pointToTuple(points.part(i)), 1, colorGreen, thickness=1)

    return image, imageCoordinates

def drawTriangles(fileName):
    image = cv2.imread(fileName)

    imageMarked, imagePts = markTheHumanFace(fileName)

    subdiv = cv2.Subdiv2D((0,0,image.shape[1],image.shape[0]))

    for i in range(68):
        subdiv.insert(imagePts[i])

    height = image.shape[0]
    width = image.shape[1]

    subdiv.insert((0, 0))
    subdiv.insert((width / 2, 0))
    subdiv.insert((width - 1, 0))
    subdiv.insert((width - 1, height / 2))
    subdiv.insert((width - 1, height - 1))
    subdiv.insert((width / 2, height - 1))
    subdiv.insert((0, height - 1))
    subdiv.insert((0, height / 2))

    triangles = subdiv.getTriangleList()

    indexedPoints = []

    imagePts.append((0, 0))
    imagePts.append((int(width / 2), 0))
    imagePts.append((width - 1, 0))
    imagePts.append((width - 1, int(height / 2)))
    imagePts.append((width - 1, height - 1))
    imagePts.append((int(width / 2), height - 1))
    imagePts.append((0, height - 1))
    imagePts.append((0, height / 2))

    print(triangles[0])
    for i in range(len(triangles)):
        zero = triangles[i][0]
        one = triangles[i][1]
        two = triangles[i][2]
        three = triangles[i][3]
        four = triangles[i][4]
        five = triangles[i][5]

        firstCoord = (int(zero), int(one))
        secondCoord = (int(two), int(three))
        thirdCoord = (int(four), int(five))

        indexedPoints.append([pointToIndex(firstCoord, imagePts),
                              pointToIndex(secondCoord, imagePts),
                              pointToIndex(thirdCoord, imagePts)])

    return indexedPoints

""""
    for i in range(len(triangles)):
        sel_triangle = triangles[i].astype(np.int)
        cv2.line(image, (sel_triangle[0], sel_triangle[1]), (sel_triangle[2], sel_triangle[3]), colorGreen)
        cv2.line(image, (sel_triangle[0], sel_triangle[1]), (sel_triangle[4], sel_triangle[5]), colorGreen)
        cv2.line(image, (sel_triangle[2], sel_triangle[3]), (sel_triangle[4], sel_triangle[5]), colorGreen)
"""


def pointToTuple(point):
    temp = str(point)
    splitted = temp.split(",")
    coordX = splitted[0][1:]
    coordY = splitted[1][1:]
    coordY = coordY[:-1]
    return (int(coordX), int(coordY))

def imageToPoints(fileName, imageType):
    if("cat" in imageType):
        image = cv2.imread(fileName)

        pointsArray = []
        height = image.shape[0]
        width = image.shape[1]

        pointsArray.append((0, 0))
        pointsArray.append((width / 2, 0))
        pointsArray.append((width - 1, 0))
        pointsArray.append((width - 1, height / 2))
        pointsArray.append((width - 1, height - 1))
        pointsArray.append((width / 2, height - 1))
        pointsArray.append((0, height - 1))
        pointsArray.append((0, height / 2))

        imagePts = part2.catToPoints(fileName)
        for i in range(len(imagePts)):
            pointsArray.append(imagePts[i])

        return pointsArray


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("inputs/shape_predictor_68_face_landmarks.dat")

    image = cv2.imread(fileName)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height = image.shape[0]
    width = image.shape[1]

    rectangles = detector(gray)
    points = predictor(gray, rectangles[0])

    pointsArray = []

    height = image.shape[0]
    width = image.shape[1]

    pointsArray.append((0, 0))
    pointsArray.append((width / 2, 0))
    pointsArray.append((width - 1, 0))
    pointsArray.append((width - 1, height / 2))
    pointsArray.append((width - 1, height - 1))
    pointsArray.append((width / 2, height - 1))
    pointsArray.append((0, height - 1))
    pointsArray.append((0, height / 2))
    for i in range(68):
       pointsArray.append(pointToTuple(points.part(i)))

    return pointsArray

def applyAffineTransform(source, sourceTriangular, destinationTriangular, shapes,):

    warpedMatrix = cv2.getAffineTransform(np.float32(sourceTriangular), np.float32(destinationTriangular))
    return cv2.warpAffine(source, warpedMatrix, (shapes[0], shapes[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def applyMorphToTriang(image1, image2, image, t1, t2, t, alpha):

    t1Rect = []
    t2Rect = []
    tRect = []

    rect1 = cv2.boundingRect(np.float32(t1))
    rect2 = cv2.boundingRect(np.float32(t2))
    rect = cv2.boundingRect(np.float32(t))

    for i in range(3):
        tRect.append(((t[i][0] - rect[0]), (t[i][1] - rect[1])))
        t1Rect.append(((t1[i][0] - rect1[0]), (t1[i][1] - rect1[1])))
        t2Rect.append(((t2[i][0] - rect2[0]), (t2[i][1] - rect2[1])))

    maskedImage = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(maskedImage, np.int32(tRect), (1.0,1.0,1.0), 16, 0)

    image1Rect = image1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0]+rect1[2]]
    image2Rect = image2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0]+rect2[2]]

    sizeOfRect = (rect[2], rect[3])
    warpedImage1 = applyAffineTransform(image1Rect, t1Rect, tRect, sizeOfRect)
    warpedImage2 = applyAffineTransform(image2Rect, t2Rect, tRect, sizeOfRect)

    imgRect = (1.0 - alpha) * warpedImage1 + alpha * warpedImage2

    image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] * ( 1 - maskedImage ) + imgRect * maskedImage

if __name__== "__main__":
    alphaChange = []

    for alpha in range(0, 10):
        alpha *= 0.1
        filename1 = "inputs/input4.png"
        filename2 = "inputs/input3.png"

        img1 = cv2.imread(filename1)
        img2 = cv2.imread(filename2)

        img1 = np.float32(img1)
        img2 = np.float32(img2)

        points1 = imageToPoints(filename1, "human")
        points2 = imageToPoints(filename2, "human")

        points = []

        for i in range(0, len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((int(x), int(y)))

        imgMorph = np.zeros(img1.shape, dtype=img1.dtype)

        indexedPoints1 = drawTriangles("inputs/input4.png")

        for i in range(len(indexedPoints1)):
            x = indexedPoints1[i][0]
            y = indexedPoints1[i][1]
            z = indexedPoints1[i][2]

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]

            applyMorphToTriang(img1, img2, imgMorph, t1, t2, t, alpha)

        #cv2.imshow("olsun Allah'im", np.uint8(imgMorph))
        #cv2.waitKey()
        temp = cv2.cvtColor(np.uint8(imgMorph), cv2.COLOR_BGR2RGB)
        alphaChange.append(temp)

    clip = ImageSequenceClip(alphaChange, fps=1)
    clip.write_videofile('outputs/part5_Result_Video4' + '.mp4', codec="mpeg4")






