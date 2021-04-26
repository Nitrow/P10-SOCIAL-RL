#!/usr/bin/env python
"""experiment controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor, Display, MouseState, Mouse
import numpy as np
import cv2

# If you want the camera image
cam = True

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
#display = supervisor.getDevice("display_robot")
display_score = supervisor.getDevice("display_score")
display_score.setOpacity(0)
#display_score.drawLine(0, 100, 10, 50)  # Test if drawing a line works (yes, it does)
#display_score.drawText("Hello", 0, 3)
width = camera.getWidth()
height = camera.getHeight()
# mouse = Mouse()
# mouse.enable(timestep)
# mouse.enable3dPosition()

selection = None
prevSelection = None
canSelection = None
can_height = 0.85
selectionName = None
canSelectionName = None

missed = 0
correctSort = 0
wrongSort = 0

# Get can objects
cans = []
root_children = supervisor.getRoot().getField("children")
root_children_n = root_children.getCount()
for n in range(root_children_n):
    if "CAN" in root_children.getMFNode(n).getDef():
        cans.append(root_children.getMFNode(n).getId())


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
    
    if cameraData:
        ir = display_score.imageNew(cameraData, Display.BGRA, camera.getWidth(), camera.getHeight())
        display_score.imagePaste(ir, 0, 0, False)
        display_score.imageDelete(ir)
    #imageRef = display.imageNew(cameraData, Display.ARGB, camera.getHeight(), camera.getWidth())
    #display.imagePaste(imageRef, 1024, 768)        
    cv2.imshow("preview", image)
    cv2.waitKey(timestep)

while supervisor.step(timestep) != -1:

    selection = supervisor.getSelected()
    selectionName = selection.getDef() if selection else ""
    selectionColor = selectionName.split('_')[0]

    if "CAN" in selectionName:
        canSelection = selection
        canColor = selectionColor
        
    elif ("CRATE" in selectionName) and canSelection:
        new_position = selection.getField("translation").getSFVec3f()
        new_position[1] = can_height
        canSelection.getField("translation").setSFVec3f(new_position)
        if selectionColor == canColor:
            correctSort += 1
            cans.remove(canSelection.getId())
        else: wrongSort += 1 
        canSelection = None

    # Check for missed ones:
    missCount = 0
    for canID in cans:  
        canX, _, _ = supervisor.getFromId(canID).getField("translation").getSFVec3f()
        if canX < -1.5:
            missCount += 1

    missed = missCount
    prevSelection = selection
    #print("Correct: {}\t Incorrect: {}\t Missed: {}\t Total: {}".format(correctSort, wrongSort, missed, correctSort-wrongSort-missed))
    
    if cam: drawImage(camera)