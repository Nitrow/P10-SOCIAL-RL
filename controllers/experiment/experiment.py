#!/usr/bin/env python3
"""experiment controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor, Display, MouseState, Mouse
import numpy as np
import cv2
# create the Robot instance.
#robot = Robot()
supervisor = Supervisor()
robot = supervisor.getFromDef("UR3")
# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())
cv2.startWindowThread()
cv2.namedWindow("preview")
padding = np.array([10, 10])
camera = Camera("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)
display = Display("display")

width = camera.getWidth()
height = camera.getHeight()
mouse = Mouse()
mouse.enable(timestep)
mouse.enable3dPosition()

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller

def drawImage(camera):
    cameraData = camera.getImage()
    image = np.frombuffer(cameraData, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    for object in camera.getRecognitionObjects():
        size = np.array(object.get_size_on_image()) + padding
        start_point = np.array(object.get_position_on_image()) - (size / 2) 
        start_point =  np.array([int(x) for x in start_point])
        end_point = start_point + size
        color = np.rint(np.array(object.get_colors())*255)
        thickness = 2
        #color = [0, 255, 0]
        color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
        image = cv2.rectangle(image, tuple(start_point), tuple(end_point), tuple(color), thickness)        
    cv2.imshow("preview", image)
    cv2.waitKey(timestep)
       

prevClick = False
doubleClick = False
selection = None
prevSelection = None
canSelection = None
can_height = 0.85
while supervisor.step(timestep) != -1:
    event = mouse.getState()
    click = event.left
    #print(click)
    if (click and not prevClick):
        selection = supervisor.getSelected()
        if selection == prevSelection: doubleClick = True
        selectionName = selection.getDef()
        
        if ("CAN" in selectionName):
            canSelection = selection.getField("translation")
            
        elif ("CRATE" in selectionName) and canSelection:
            new_position = selection.getField("translation").getSFVec3f()
            new_position[1] = can_height
            if doubleClick:
                canSelection.setSFVec3f(new_position)
                canSelection = None
    doubleClick = False
    prevSelection = selection
    prevClick = click
    if prevSelection != prevSelection and selection != selection:
        print("Current selection: {} \t Previous selection: {}".format(selection.getDef(),prevSelection.getDef()))
    drawImage(camera)


# Enter here exit cleanup code.
