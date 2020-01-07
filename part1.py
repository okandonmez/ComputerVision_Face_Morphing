#Okan Dönmez
#150170708

import cv2
import dlib

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

    cv2.rectangle(image, rectangleCoords[0], rectangleCoords[1], colorBlue, 1)

    for i in range(68):
       cv2.circle(image, pointToTuple(points.part(i)), 1, colorGreen, thickness=1)

    return image

def catFileToCoordinates(catFileName):
    with open(catFileName, 'r') as file:
        data = file.read().replace('\n', '')

    chars = data.split(" ")
    numOfPoints = int(chars[0])

    listOfPoints = []
    for i in range(1, numOfPoints*2, 2):
        listOfPoints.append((int(chars[i]), int(chars[i+1])))

    return listOfPoints

def markTheCatFace(fileName):
    image = cv2.imread(fileName)
    listOfPoints = catFileToCoordinates(fileName + ".cat")

    for coordinate in listOfPoints:
        cv2.circle(image, coordinate, 1, (0, 255, 0))

    return image


if __name__== "__main__":
    input2 = markTheCatFace("inputs/cat.jpg")
    input3 = markTheHumanFace("inputs/input3.png")
    input4 = markTheHumanFace("inputs/input4.png")

    cv2.imshow("Human1", input3)
    cv2.imshow("Human2", input4)
    cv2.imshow("cat", input2)

    cv2.imwrite('outputs/part1_Cat_Marked.png', input2)
    cv2.imwrite('outputs/part1_Human1_Marked.png', input3)
    cv2.imwrite('outputs/part1_Human2_Marked.png', input4)

    cv2.waitKey()