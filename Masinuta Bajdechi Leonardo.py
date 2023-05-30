import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("CarRacing-v2", render_mode="human")

observation = env.reset()[0]
test = 3
action = env.action_space.sample()
initialRoadCenterCoordX = 48
initialRoadCenterCoordY = 48
previousAction = action
actualRoadCenterCoordX = 48
actualRoadCenterCoordY = 48

def getRoadCenterValue(observation):
    # Conversia in grayscale (Observation To GrayScale)
    gray = cv2.cvtColor(observation[0:84, :, :], cv2.COLOR_RGB2GRAY)

    # Aplicarea unui Gaussian Blur pentru reducerea noise-ului din imagine
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # Aplicarea unui thresholding pentru a crea imagini binare
    _, thresh = cv2.threshold(blurred, 40,40, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inchiderea "golurilor" in imaginea binara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Aplicarea unui "Canny" pentru detectarea marginilor
    canny = cv2.Canny(closed, 50,150)

    # Dilatarea imaginii pentru a umple golurile
    dilated = cv2.dilate(thresh, canny, iterations=2)

    # Gasirea conturului imaginii binare
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculul mijlocului drumului
    nz = cv2.findNonZero(canny)
    try:
        middle = (nz[:, 0, 0].max() + nz[:, 0, 0].min()) / 2
        print(f'Road middle value: {middle}')
    except:
        print('Error 404: Road Middle Not Found')
    return middle, 1

def takeAction(actualRoadCenterCoordX, carPositionCoordX, maxSteeringAngleValue, maxAccelerationValue):
    # Calcularea unghiului de virare (cu aproximare) pe baza valoare mijlocului drumului si a pozitii masinii
    steeringAngleAproximation = (actualRoadCenterCoordX - carPositionCoordX) / carPositionCoordX

    # "Cliparea" valorii minime si maxime ale unghiului de virare in intervalul aproximat
    steeringAngleValue = np.clip(steeringAngleAproximation, -maxSteeringAngleValue, maxSteeringAngleValue)

    # Setarea accelerarii la valoarea maxima
    acceleration = maxAccelerationValue

    # Setarea valorii franei la 0.0
    brake = 0.0

    # Returnarea calculului ActionSpace-ului
    return [steeringAngleValue, acceleration, brake]

while True:
    test+=1
    carPositionCoordX = 48
    maxSteeringAngle = 0.8
    maxAcceleration = 0.005
    try:
        actualRoadCenterCoordX, actualRoadCenterCoordY = getRoadCenterValue(observation)
    except:
        print('Error: Road middle could not be detected')
    if carPositionCoordX != None and actualRoadCenterCoordX != None:
       action = takeAction(actualRoadCenterCoordX, carPositionCoordX, maxSteeringAngle, maxAcceleration)
       previousAction = action
       observation, reward, done, terminated, info = env.step(action)
    else:
        action = env.action_space.sample()
        observation, reward, done, terminated, info = env.step(previousAction)
    if done:
        break

