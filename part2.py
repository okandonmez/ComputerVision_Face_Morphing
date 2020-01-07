#Okan Dönmez
#150170708

import numpy as np
import cv2

blank_image = np.zeros((500, 500, 3), np.uint8)
blank_image[:,0:500] = (0,0,0)      # (B, G, R)
catImage = cv2.imread("inputs/cat.jpg")

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

def catToPoints(fileName, isShow = False):
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

    cv2.circle(catImage, catLftEyeCntr, 2, (255, 0, 0), thickness=3)
    cv2.circle(catImage, catRghtEyeCntr, 2, (255, 0, 0), thickness=3)
    cv2.circle(catImage, catMouthCntr, 2, (255, 0, 0), thickness=3)

    if(isShow):
        cv2.imshow("Points", blank_image)
        cv2.imshow("Points transformed for the cat face", catImage)
        cv2.imwrite("outputs/part2_PointsTransformedTo_Cat_Face.png", catImage)
        cv2.imwrite("outputs/part2_Template_Points.png", catImage)
        cv2.waitKey()

    return tmpPts



if __name__== "__main__":
    catToPoints("inputs/cat.jpg", True)