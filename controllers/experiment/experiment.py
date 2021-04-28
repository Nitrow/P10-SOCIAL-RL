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

random.seed(1)


global can_num
can_num = 0

max_cans = 20

# def fileChanger(textToSearch, textToReplace):
#     for file in os.listdir("resources"):
#         f = "resources/" + file
#         filereplace(f, textToSearch, textToReplace)

#fileChanger("translation 4.05", "translation 3.1")

def moveFingers(fingers, mode="open"):

    if mode == "open":
        fingers[0].setPosition(0.04)
        fingers[1].setPosition(0.04)
    elif mode == "close":
        fingers[0].setPosition(0)
        fingers[1].setPosition(0)


def pickTargers(total_cans, choices=5):
    """
    Gets five targets based on 3 criteria (assessing each can):
    1. Right color
    2. Furthest on the conveyor belt
    3. Right pose (graspable)
    4. Closeness to other candidate

    Returns:
        A dictionary with the IDs, where each ID contains:
            1. Perceived colour of the can
            2. Position of the can
            3. Additional text (explanation - if applicable)
    """

    # Check if yellow or green
    candidates = {}
    top5_dists = []
    top5_keys = []
    for key, val in total_cans.items():
        reason = ""
        candidates[key] = [val]
        candidates[key].append(supervisor.getFromId(key).getField("translation").getSFVec3f())
        
        if val in ["green", "red"]:
            if abs(supervisor.getFromId(key).getField("rotation").getSFRotation()[3]) > 0.1:
                reason += "graspError"
            elif candidates[key][1][0] > 0.5:
                top5_keys.append(key)
                top5_dists.append(candidates[key][1][0])
        else:
            reason += "colorError"

        candidates[key].append(reason)
    top5 = sorted(zip(top5_dists, top5_keys), key=lambda x: x[1])[:choices]

    for i in range(len(top5)):
        candidates[top5[i][1]][2] = str(i+1)

    return candidates


def position_Checker():
    """
    Returns true if all joints are 0.02 rad within the desired angles
    """
    return all([abs(sensors[i].getValue() - joints[i+1]) < 0.02 for i in range(len(sensors))])


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
    motors[i].setVelocity(3.14)                
motors[0].setVelocity(1.5)         
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

candidates = {}

moveFingers(fingers) 


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
                    trueColor = can.getDef().split('_')[0].lower()
                    if random.random() <= 0.7:
                        total_cans[can_id] = trueColor
                    else:
                        options = ["yellow", "red", "green"]
                        options.remove(trueColor)
                        total_cans[can_id] = random.choice(options)
                        #total_cans.append(can.getId())
    for item in toRemove:
        item.remove()

    # for keys, vals in candidates.items():
    #   print (keys, vals)
    # print("------------------------------------")
    print(can_num)
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


def drawImage(camera, colors, candidates):
    """
    Displays the image either in a new window, or on the Display
    """
    cameraData = camera.getImage()
    image = np.frombuffer(cameraData, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    for obj in camera.getRecognitionObjects():
        # Get the id of the object
        obj_id = obj.get_id()
        reason = candidates[obj_id][2]
        # Check if the object is on the conveyor or not
        _, _, obj_z = supervisor.getFromId(obj_id).getField("translation").getSFVec3f()
        if obj_z > 0.6 or obj_z < 0.4 or reason == "":
            continue 
        # Assign color
        color = colors[candidates[obj_id][0]]
        #print(color)
        color = np.rint(np.array(color)*255)
        size = np.array(obj.get_size_on_image()) + padding
        start_point = np.array(obj.get_position_on_image()) - (size / 2) 
        start_point =  np.array([int(n) for n in start_point])
        end_point = start_point + size
        #color = np.rint(np.array(obj.get_colors())*255)
        thickness = 2
        #color = [0, 255, 0]

        color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        if reason.isdigit():
            image = cv2.rectangle(image, tuple(start_point), tuple(end_point), tuple(color), thickness)
            start_point[1] -= 20
            text = reason
        if reason == 'colorError' or reason == 'graspError':
            text = "Can't sort color" if reason == 'colorError' else 'Unable to grasp'
        cv2.putText(image, text, tuple(start_point), font, 1, tuple(color), 2)


    if cameraData:
        # Displaying the camera image directly
        #ir = display_score.imageNew(cameraData, Display.BGRA, camera.getWidth(), camera.getHeight())
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
    global can_num 
    can_num += 1
    can_distances = ["000", "999", "555", "535", "515", "495", "475", "455"]
    can_colors = ["green", "yellow", "red"]
    can = "resources/" + random.choice(can_colors) + "_can_" + random.choice(can_distances) + ".wbo"
    root_children.importMFNode(-1, can)


def endGame():
    displayScore(display_score, correctSort, wrongSort, missed)
    supervisor.step(64)
    supervisor.simulationSetMode(0)


while supervisor.step(timestep) != -1:
    total_cans, missed = countCans(missed, total_cans)
    candidates = pickTargers(total_cans, 3)
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
        if selectionColor == canColor: correctSort += 1
        else: wrongSort += 1 
        del total_cans[canSelection.getId()]
        canSelection = None

    # Check for missed ones:
    for canID in total_cans:  
        canX, _, _ = supervisor.getFromId(canID).getField("translation").getSFVec3f()
        if canX < -1.5:
            missed += 1

    prevSelection = selection
    
    if random.randrange(0,100) % 50 == 0:
        if can_num < max_cans:
            generateCans()
        #pass
    #print("Correct: {}\t Incorrect: {}\t Missed: {}\t Total: {}".format(correctSort, wrongSort, missed, correctSort-wrongSort-missed))
    #print(onConveyorRanked(total_cans))
    if cam: drawImage(camera, colors, candidates)
    if can_num >= max_cans and not bool(total_cans):
        endGame()

####################################################################################################################### 
#######################################################################################################################
#######################################################################################################################

    # if prepare_grasp == True and bool(total_cans):

    #      index = list(total_cans.keys())[0] #####SETTING THE CAN, CAN BE REPALCED BY AN ACTUAL ID#####
    #      goal = supervisor.getFromId(index).getField("translation")
    #      target = np.array(goal.getSFVec3f())

    #      target_position = [target[2], 0.167, target[1]-0.48]
    #      #target_position = [target[2], 0.167, target[1]]
    
    #      orientation_axis = "Y"
    #      target_orientation = [0, 0, -1]
        
    #      joints = my_chain.inverse_kinematics(target_position, target_orientation=target_orientation, orientation_mode=orientation_axis)
        
    #      for i in range(len(joint_names)):
    #          motors[i].setPosition(joints[i+1])    
    #      print(joints)
    #      #prepare_grasp = not position_Checker()
    #      #print(distance_sensor.getValue())
    #      prepare_grasp = False

    # if  prepare_grasp == False and position_Checker()==True and distance_sensor.getValue() < 800 and target[0] < 0.19 :
         
    #      for i in range(len(joint_names)):
    #          motors[1].setPosition(0.15)    
       
    #      prepare_grap2 = True        

    # if  prepare_grap2 == True and distance_sensor.getValue() < 200:
        
    #      motors[1].setPosition(sensors[1].getValue())

    #      moveFingers(fingers, "close")

    #      go_to_bucket = True        
    #      prepare_grap2 = False

    # if  go_to_bucket == True and go_to_bucket2 == False and sensor_fingers[0].getValue() < 0.012 or sensor_fingers[1].getValue() < 0.012:

    #      for i in range(len(joint_names)):
    #              motors[1].setPosition(-1)
         
    #      if sensors[1].getValue()-0.2 < -1 < sensors[1].getValue()+0.2:
    #              go_to_bucket2 = True    
    #              go_to_bucket = False
    
    # if go_to_bucket2 == True:
         
    #       if index > 5:
    #           for i in range(len(joint_names)):
    #                   motors[0].setPosition(1.5)
    #                   if sensors[0].getValue()-0.01 < 1.5 < sensors[0].getValue()+0.01:
    #                       drop = True
    #                       go_to_bucket2 = False

    #       if index < 3:
    #           for i in range(len(joint_names)):
    #                   motors[0].setPosition(-1.8)
    #                   if sensors[0].getValue()-0.01 < -1.8 < sensors[0].getValue()+0.01:
    #                       drop = True
    #                       go_to_bucket2 = False

         
    # if drop == True:
    #     moveFingers(fingers, mode = "open") 
       
    #     if sensor_fingers[0].getValue()-0.005 < 0.03 < sensor_fingers[0].getValue()+0.005: 
    #         #prepare_grasp = True
    #         drop = False
    # if bool(total_cans) and not prepare_grap2: 
    #      goal = supervisor.getFromId(index).getField("translation")
    #      target = np.array(goal.getSFVec3f())          
    
