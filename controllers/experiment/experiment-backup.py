#!/usr/bin/env python
"""experiment controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor, Display, MouseState, Mouse
import numpy as np
import cv2
import random
import os
from pyutil import filereplace



# def fileChanger(textToSearch, textToReplace):
#     for file in os.listdir("resources"):
#         f = "resources/" + file
#         filereplace(f, textToSearch, textToReplace)

#fileChanger("translation 4.05", "translation 3.1")





random.seed(1)

y = 0.88
x = 3.17

# If you want the camera image
cam = True

supervisor = Supervisor()
robot = supervisor.getFromDef("UR3")

# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())
#if cam: 
#    cv2.startWindowThread()
#    cv2.namedWindow("preview")
padding = np.array([10, 10])
camera = Camera("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)
#display = supervisor.getDevice("display_robot")
display_explanation = supervisor.getDevice("display_explanation")
display_explanation.setOpacity(0)

display_score = supervisor.getDevice("display_score")
display_score.setOpacity(1)
display_score.setAlpha(0)
display_score.fillRectangle(0, 0, display_score.getWidth(), display_score.getHeight())
display_score.setAlpha(1)
display_score.setColor(0x000000)
display_score.setFont("Lucida Console", 64, True)
display_score.drawText("Scoring display", 0, 0)
display_score.fillRectangle(0, 100, 800, 10)
#display_explanation.drawLine(0, 100, 10, 50)  # Test if drawing a line works (yes, it does)
#display_explanation.drawText("Hello", 0, 3)
width = camera.getWidth()
height = camera.getHeight()
# mouse = Mouse()
# mouse.enable(timestep)
# mouse.enable3dPosition()
colors = {"yellow" : [0.309804, 0.913725, 1.0], "red" : [0.0, 0.0, 1.0], "green" : [0.0, 1.0, 0.0]}

selection = None
prevSelection = None
canSelection = None
can_height = 0.85
selectionName = None
canSelectionName = None

can_num = 1
missed = 0
correctSort = 0
wrongSort = 0

# Get can objects

root_children = supervisor.getRoot().getField("children")


def displayScore(display, correct, incorrect, missed):
    h = int(display.getHeight() / 5)
    x = h

    display.setAlpha(0)
    display.fillRectangle(0, h, display.getWidth(), display.getHeight())
    display.setAlpha(1)
    display.setFont("Lucida Console", 64, True)
    display.drawText("Correct:   {}".format(correct), 10, x)
    x += h
    display.drawText("Incorrect: {}".format(incorrect), 10, x)
    x += h
    display.drawText("Missed:    {}".format(missed), 10, x)
    x += h
    #display.setFont("Lucida Console", 48, True)
    display.drawText("Total:     {}".format(correct-incorrect-missed), 10, x)


def countCans(missed):
    total_cans = {}
    toRemove = []
    root_children_n = root_children.getCount()
    for n in range(root_children_n):
        if "CAN" in root_children.getMFNode(n).getDef():
            can = root_children.getMFNode(n)
            x, y, z = can.getField("translation").getSFVec3f()
            if not x < -1.2:# and y >= 0.8:
            #root_children.getMFNode(n).remove()
                if y >= 0.8:
                    if random.random() <= 0.6:
                        total_cans[can.getId()] = can.getDef().split('_')[0].lower()
                    else:
                        total_cans[can.getId()] = random.choice(["yellow", "red", "green"])
                    #total_cans.append(can.getId())
            else:
                missed += 1
                toRemove.append(can)
    for item in toRemove:
        item.remove()
    return total_cans, missed


def onConveyorRanked(cans):
    """
    Returns a ranked list of the cans that are on the conveyor belt
    """
    cansOnConveyor = cans[:]
    for canID in cans:  
        canX, canY, _ = supervisor.getFromId(canID).getField("translation").getSFVec3f()
        if canY <= 0.8 or canID == supervisor.getSelected().getId():
            cansOnConveyor.remove(canID)
    return cansOnConveyor


def drawImage(camera, colors):
    """
    Displays the image either in a new window, or on the Display
    """
    cameraData = camera.getImage()
    image = np.frombuffer(cameraData, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    for obj in camera.getRecognitionObjects():
        # Get the id of the object
        obj_id = obj.get_id()
        # Assign color with 40% error rate
        if random.random() <= 0.6:
            color = np.rint(np.array(obj.get_colors())*255)
            obj_color = list(colors.keys())[list(colors.values()).index(obj.get_colors())]
        else:
            color = np.rint(np.array(random.choice(list(colors.values())))*255)
            obj_color = random.choice(list(colors.values()))
        size = np.array(obj.get_size_on_image()) + padding
        start_point = np.array(obj.get_position_on_image()) - (size / 2) 
        start_point =  np.array([int(n) for n in start_point])
        end_point = start_point + size
        thickness = 2
        #color = [0, 255, 0]
        color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
        image = cv2.rectangle(image, tuple(start_point), tuple(end_point), tuple(color), thickness)
    
    if cameraData:
        # Displaying the camera image directly
        #ir = display_explanation.imageNew(cameraData, Display.BGRA, camera.getWidth(), camera.getHeight())
        # Displaying the processed image
        cv2.imwrite('tmp.jpg', image)
        ir = display_explanation.imageLoad('tmp.jpg')
        display_explanation.imagePaste(ir, 0, 0, False)
        display_explanation.imageDelete(ir)
    #imageRef = display.imageNew(cameraData, Display.ARGB, camera.getHeight(), camera.getWidth())
    #display.imagePaste(imageRef, 1024, 768)        
    #cv2.imshow("preview", image)
    #cv2.waitKey(timestep)


def generateCans():
    
    # 
    # Set coordinates, y = 0.88  # Height of the conveyor belt
    #x += random.uniform(0.06, 0.1)
    #z = random.uniform(0.455,0.555) # 0.455 < z < 0.555 - Width of the conveyor belt

    can_distances = [555, 535, 515, 495, 475, 455]
    can_colors = ["green", "yellow", "red"]
    can = "resources/" + random.choice(can_colors) + "_can_" + str(random.choice(can_distances)) + ".wbo"
    root_children.importMFNode(-1, can)


while supervisor.step(timestep) != -1:
    total_cans, missed = countCans(missed)
    #print(total_cans)
    selection = supervisor.getSelected()
    selectionName = selection.getDef() if selection else ""
    selectionColor = selectionName.split('_')[0]
    for keys, vals in total_cans.items():
        print (keys, vals)
    if "CAN" in selectionName:
        canSelection = selection
        canColor = selectionColor
        
    elif ("CRATE" in selectionName) and canSelection:
        new_position = selection.getField("translation").getSFVec3f()
        new_position[1] = can_height
        canSelection.getField("translation").setSFVec3f(new_position)
        if selectionColor == canColor:
            correctSort += 1
            del total_cans[canSelection.getId()]
        else: wrongSort += 1 
        canSelection = None

    # # Check for missed ones:
    # for canID in total_cans:  
    #     canX, _, _ = supervisor.getFromId(canID).getField("translation").getSFVec3f()
    #     if canX < -1.5:
    #         missed += 1

    prevSelection = selection
    displayScore(display_score, correctSort, wrongSort, missed)
    if random.randrange(0,100) % 50 == 0:
        generateCans()
        #pass
    #print("Correct: {}\t Incorrect: {}\t Missed: {}\t Total: {}".format(correctSort, wrongSort, missed, correctSort-wrongSort-missed))
    #print(onConveyorRanked(total_cans))
    if cam: drawImage(camera, colors)