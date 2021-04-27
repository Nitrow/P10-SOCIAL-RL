#!/usr/bin/env python3
"""experiment controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor, Display, MouseState, Mouse
import numpy as np
import cv2
import random
import os
import ikpy
from ikpy.chain import Chain



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

total_cans = {}

can_num = 1
missed = 0
correctSort = 0
wrongSort = 0



joint_names = ['shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint']
        
finger_names = ['right_finger', 'left_finger']                
motors = [0] * len(joint_names)
sensors = [0] * len(joint_names)
fingers = [0] * len(finger_names)
sensor_fingers = [0] * len(finger_names)

for i in range(len(joint_names)):  
    motors[i] = supervisor.getDevice(joint_names[i])   
    
    sensors[i] = supervisor.getDevice(joint_names[i] + '_sensor')
    sensors[i].enable(timestep)
    #motors[i].setPosition(float('inf'))
    #motors[i].setVelocity(3.14)                
#motors[0].setVelocity(1.5)         
for i in range(len(finger_names)):  
    fingers[i] = supervisor.getDevice(finger_names[i])
    sensor_fingers[i] = supervisor.getDevice(finger_names[i] + '_sensor')
    sensor_fingers[i].enable(timestep)

distance_sensor = supervisor.getDevice("distance_sensor1") 
distance_sensor.enable(timestep)  

my_chain = ikpy.chain.Chain.from_urdf_file("resources/robot2.urdf")      


prepare_grasp = True
go_to_bucket = False
prepare_grap2 = False
go_to_bucket2 = False
drop = False

fingers[0].setPosition(0.04)
fingers[1].setPosition(0.04)


def position_Checker():
    if sensors[0].getValue()-0.02 < joints[1] < sensors[0].getValue()+0.02 and sensors[1].getValue()-0.02 < joints[2] < sensors[1].getValue()+0.02 and sensors[2].getValue()-0.02 < joints[3] < sensors[2].getValue()+0.02 and sensors[3].getValue()-0.02 < joints[4] < sensors[3].getValue()+0.02 and sensors[4].getValue()-0.02 < joints[5] < sensors[4].getValue()+0.02 and sensors[5].getValue()-0.02 < joints[6] < sensors[5].getValue()+0.02:
        return True
    else:
        return False


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


def countCans(missed, total_cans):
    toRemove = []
    root_children_n = root_children.getCount()
    for n in range(root_children_n):
        if "CAN" in root_children.getMFNode(n).getDef():
            can = root_children.getMFNode(n)
            can_id = can.getId()
            x, y, z = can.getField("translation").getSFVec3f()
            # If we already have the can, check if it should be removed
            if can_id in list(total_cans.keys()):
                if x < -1.2:# and y >= 0.8:
                    missed += 1
                    toRemove.append(can)
                    del total_cans[can_id]
                elif z > 0.6 or z < 0.4:
                    del total_cans[can_id]
            # If the can is not in the list yet, we should add it
            else:
                if y >= 0.8:
                    if random.random() <= 0.6:
                        total_cans[can_id] = can.getDef().split('_')[0].lower()
                    else:
                        total_cans[can_id] = random.choice(["yellow", "red", "green"])
                        #total_cans.append(can.getId())
    for keys, vals in total_cans.items():
        print (keys, vals)
    for item in toRemove:
        item.remove()
    print("------------------------------------")
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


def drawImage(camera, colors, total_cans):
    """
    Displays the image either in a new window, or on the Display
    """
    cameraData = camera.getImage()
    image = np.frombuffer(cameraData, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    for obj in camera.getRecognitionObjects():
        # Get the id of the object
        obj_id = obj.get_id()
        # Check if the object is on the conveyor or not
        _, _, obj_z = supervisor.getFromId(obj_id).getField("translation").getSFVec3f()
        if obj_z > 0.6 or obj_z < 0.4:
            continue 
        # Assign color
        color = colors[total_cans[obj_id]]
        #print(color)
        color = np.rint(np.array(color)*255)
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

    can_distances = [555, 535, 515, 495, 475, 455]
    can_colors = ["green", "yellow", "red"]
    can = "resources/" + random.choice(can_colors) + "_can_" + str(random.choice(can_distances)) + ".wbo"
    root_children.importMFNode(-1, can)


while supervisor.step(timestep) != -1:
    total_cans, missed = countCans(missed, total_cans)
    #print(total_cans)
    selection = supervisor.getSelected()
    selectionName = selection.getDef() if selection else ""
    selectionColor = selectionName.split('_')[0]
    # for keys, vals in total_cans.items():
    #     print (keys, vals)
    if "CAN" in selectionName:
        canSelection = selection
        canColor = selectionColor
        
    elif ("CRATE" in selectionName) and canSelection:
        new_position = selection.getField("translation").getSFVec3f()
        new_position[1] = can_height
        canSelection.getField("translation").setSFVec3f(new_position)
        if selectionColor == canColor:
            correctSort += 1
        else:
            wrongSort += 1
        del total_cans[canSelection.getId()]
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
    if cam: drawImage(camera, colors, total_cans)