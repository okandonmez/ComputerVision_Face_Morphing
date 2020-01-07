#Okan Dönmez
#150170708

import numpy as np
import cv2
import dlib

blank_image = np.zeros((500, 500, 3), np.uint8)
blank_image[:,0:500] = (0,0,0)      # (B, G, R)
catImage = cv2.imread("inputs/cat.jpg")

def pointToTuple(point):
    temp = str(point)
    splitted = temp.split(",")
    coordX = splitted[0][1:]
    coordY = splitted[1][1:]
    coordY = coordY[:-1]
    return (int(coordX), int(coordY))

def recToCoordinates(rectangle):
    rectStr = str(rectangle)
    temp = rectStr.split(")")

    firsCoord = temp[0][2:]
    secondCoord = temp[1][2:]

    tmpCoordTop = firsCoord.split(",")
    tmpCoordBottom = secondCoord.split(",")

    topX = tmpCoordTop[0]
    topY = tmpCoordTop[1][1:]

    bottomX = tmpCoordBottom[0]
    bottomY = tmpCoordBottom[1][1:]

    rectangleCoords = []
    rectangleCoords.append((int(topX), int(topY)))
    rectangleCoords.append((int(bottomX), int(bottomY)))

    return rectangleCoords


def markTheHumanFace(fileName):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("inputs/shape_predictor_68_face_landmarks.dat")

    image = cv2.imread(fileName)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectangles = detector(gray)
    points = predictor(gray, rectangles[0])

    colorGreen = (0, 255, 0)
    colorBlue = (255, 0, 0)

    rectangleCoords = recToCoordinates(rectangles[0])

    #cv2.rectangle(image, rectangleCoords[0], rectangleCoords[1], colorBlue, 1)

    imageCoordinates = []
    for i in range(68):
        imageCoordinates.append(pointToTuple(points.part(i)))
        cv2.circle(image, pointToTuple(points.part(i)), 1, colorGreen, thickness=1)

    return image, imageCoordinates

def findMiddleOfTwoPoints(point1, point2): # Gelen 2 noktanın orta noktasını tuple olarak
    middleX = (point1[0] + point2[0])/2
    middleY = (point1[1] + point2[1])/2

    return (int(middleX), int(middleY))


def catFileToCoordinates(catFileName):      # Cat dosyasını okuyup koordinatları liste döner
    with open(catFileName, 'r') as file:
        data = file.read().replace('\n', '')

    chars = data.split(" ")
    numOfPoints = int(chars[0])

    listOfPoints = []
    for i in range(1, numOfPoints*2, 2):
        listOfPoints.append((int(chars[i]), int(chars[i+1])))

    return listOfPoints


def markTheCatFace(fileName):       # Cat resmine noktaları işaretler
    image = cv2.imread(fileName)
    listOfPoints = catFileToCoordinates(fileName + ".cat")

    for coordinate in listOfPoints:
        cv2.circle(image, coordinate, 1, (0, 255, 0), 4)

    return image

def tempPtsToCatImage(fileName):
    templatePoints = np.load("inputs/template_points.npy")

    catImageName = fileName
    catFileName = catImageName + ".cat"

    catPts = catFileToCoordinates(catFileName)
    tmpPts = []

    for i in range(68):  # Tmplate içindeki noktarları insani yerlerine yerleştirdim
        tmpPts.append((templatePoints[i][1], templatePoints[i][0] - 200))

    for i in range(68):  # Boş resimde 68noktayı işaretledim
        cv2.circle(blank_image, tmpPts[i], 1, (255, 255, 255), thickness=3)

    tmpLftEyeCntr = findMiddleOfTwoPoints(tmpPts[36], tmpPts[39])
    tmpRghtEyeCntr = findMiddleOfTwoPoints(tmpPts[42], tmpPts[45])
    tmpMouthCntr = tmpPts[66]

    catLftEyeCntr = catPts[0]
    catRghtEyeCntr = catPts[1]
    catMouthCntr = catPts[2]

    tmpHorizontalLength = tmpLftEyeCntr[0] - tmpRghtEyeCntr[0]
    catHorizontalLenth = catLftEyeCntr[0] - catRghtEyeCntr[0]

    tmpMiddleOfEyes = findMiddleOfTwoPoints(tmpLftEyeCntr, tmpRghtEyeCntr)
    catMiddleOfEyes = findMiddleOfTwoPoints(catLftEyeCntr, catRghtEyeCntr)

    tmpVerticalLength = tmpMiddleOfEyes[1] - tmpMouthCntr[1]
    catVerticalLenth = catMiddleOfEyes[1] - catMouthCntr[1]

    horizontalRatio = catHorizontalLenth / tmpHorizontalLength
    verticalRatio = catVerticalLenth / tmpVerticalLength

    for i in range(68):  # templatein dudağıyla catin dudağını eşitledim
        tmpPts[i] = (tmpPts[i][0] - 63, tmpPts[i][1] - 41)

    catImage = markTheCatFace(catImageName)  # cat'i işaretledim

    for i in range(68):  # her nokta için yeni koordinat hesabı
        horizontalDifference = tmpPts[66][0] - tmpPts[i][0]
        verticalDifference = tmpPts[66][1] - tmpPts[i][1]

        newHorizontalValue = horizontalDifference * horizontalRatio
        newVerticalValue = verticalDifference * verticalRatio

        tmpPts[i] = (int(tmpPts[66][0] - newHorizontalValue), int(tmpPts[66][1] - newVerticalValue))

    for i in range(68):  # bulunan yeni noktaların işaretlenmesi
        cv2.circle(catImage, tmpPts[i], 1, (255, 0, 0), 4)

    cv2.circle(blank_image, tmpLftEyeCntr, 2, (255, 0, 0), thickness=3)
    cv2.circle(blank_image, tmpRghtEyeCntr, 2, (255, 0, 0), thickness=3)
    cv2.circle(blank_image, tmpMouthCntr, 2, (255, 0, 0), thickness=3)

    #cv2.circle(catImage, catLftEyeCntr, 2, (255, 0, 0), thickness=3)
    #cv2.circle(catImage, catRghtEyeCntr, 2, (255, 0, 0), thickness=3)
    #cv2.circle(catImage, catMouthCntr, 2, (255, 0, 0), thickness=3)

    #cv2.imshow("aksd", blank_image)
    #cv2.waitKey()

    #cv2.imshow("kajsd", catImage)
    #cv2.waitKey()

    return catImage, tmpPts

def drawTriangles(fileName, imageType):
    image = cv2.imread(fileName)

    if("cat" in imageType):
        imageMarked, imagePts = tempPtsToCatImage(fileName)
    else:
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
    colorGreen = (0, 255, 0)

    for i in range(len(triangles)):
        sel_triangle = triangles[i].astype(np.int)
        cv2.line(image, (sel_triangle[0], sel_triangle[1]), (sel_triangle[2], sel_triangle[3]), colorGreen)
        cv2.line(image, (sel_triangle[0], sel_triangle[1]), (sel_triangle[4], sel_triangle[5]), colorGreen)
        cv2.line(image, (sel_triangle[2], sel_triangle[3]), (sel_triangle[4], sel_triangle[5]), colorGreen)

    return image

if __name__== "__main__":

    imageYusufHoca = drawTriangles("inputs/input1.jpg", "human")
    imageHuman1 = drawTriangles("inputs/input3.png", "human")
    imageHuman2 = drawTriangles("inputs/input4.png", "human")
    imageCat = drawTriangles("inputs/cat.jpg", "cat")

    cv2.imshow("YusufHoca", imageYusufHoca)
    cv2.imwrite("outputs/part3_Triangled_Yusuf_Hoca.png", imageYusufHoca)

    cv2.imshow("Human1", imageHuman1)
    cv2.imwrite("outputs/part3_Triangled_Redundant_American1.png", imageHuman1)

    cv2.imshow("Human2", imageHuman2)
    cv2.imwrite("outputs/part3_Triangled_Redundant_American2.png", imageHuman2)

    cv2.imshow("cat", imageCat)
    cv2.imwrite("outputs/part3_Triangled_Cat.png", imageCat)

    cv2.waitKey()
