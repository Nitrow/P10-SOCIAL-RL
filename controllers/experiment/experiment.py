#!/usr/bin/env python3
"""experiment controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor, Display, MouseState, Mouse
import numpy as np
import cv2

# If you want the camera image
cam = False

supervisor = Supervisor()
robot = supervisor.getFromDef("UR3")
# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())
if cam: 
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
selectionName = None
prevSelectionName = None

while supervisor.step(timestep) != -1:
    event = mouse.getState()
    click = event.left # False if not clicked, true while clicked
    # When we click on something (and release the button)
    if (click and not prevClick):
        # Get what we clicked on
        selection = supervisor.getSelected()
        selectionName = selection.getDef()
        
        if ("CAN" in selectionName):
            canSelection = selection.getField("translation")
            
        elif ("CRATE" in selectionName) and canSelection:
            new_position = selection.getField("translation").getSFVec3f()
            new_position[1] = can_height

    if doubleClick and canSelection:
        #canSelection.setSFVec3f(new_position)
        canSelection = None
        doubleClick = False

    if selection == prevSelection and canSelection: doubleClick = True
    
    prevSelection = selection
    prevSelectionName = selectionName
    prevClick = click
    #print("prevSelection: {}\tselection: {}\tdoubleClick: {}\t".format(prevSelectionName, selectionName, doubleClick))
    if prevSelection != prevSelection and selection != selection:
        print("Current selection: {} \t Previous selection: {}".format(selection.getDef(),prevSelection.getDef()))
    if cam: drawImage(camera)


# Enter here exit cleanup code.
