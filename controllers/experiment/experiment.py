#!/usr/bin/env python3
"""experiment controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor, Display, Connector
import numpy as np
import cv2
import random
import os
import ikpy
import math
from ikpy.chain import Chain


random.seed(10)


can_num = 0
spawn_timer = 0
spawn_limit = 20
pos_choice = "000"

reason_dict = { 'colorError' : "Can't sort color", 
                'graspError': "Unable to grasp", 
                'proximityError': "Can't reach in time"}

# 50-33 takes 100sec
#max_cans = 20  # 20 is doable with 50 freq
#freq = 50  # Less is more - 50 is doable
max_cans = 20  # 20 is doable with 50 freq
freq = 33  # Less is more - 50 is doable
# from pyutil import filereplace
# def fileChanger(textToSearch, textToReplace):
#     for file in os.listdir("resources"):
#         f = "resources/" + file
#         filereplace(f, textToSearch, textToReplace)

# fileChanger('children [', 'children [    Connector {      translation 0 0.02 0      rotation -1 0 0 -3.137485307179586      model "can_grasp"      type "passive"      axisTolerance 3.14      rotationTolerance 3.14    }')

def moveFingers(fingers, mode="open"):

    if mode == "open":
        fingers[0].setPosition(0.04)
        fingers[1].setPosition(0.04)
    elif mode == "close":
        fingers[0].setPosition(0)
        fingers[1].setPosition(0)


def getFirstCan(candidates):
    for key, val in candidates.items():
        if val[2] == '1':
            return key
    return False


def pickTargets(total_cans, choices=5, min_dist = 0.5):
    """
    Gets five targets based on 3 criteria (assessing each can):
    1. Right color
    2. Furthest on the conveyor belt
    3. Right pose (graspable)
    4. Closeness to other candidate (can't make it in time)

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
        # The can has to be closer then 0.5 and within the range of the conveyor or be (grasped above conveyor)
        if candidates[key][1][0] >= 0.5 and (int(candidates[key][1][2]*10) in range(4,6) or candidates[key][1][1] > 0.9):
            if val[1] in ["green", "red"]:
                if "ROTATED" in supervisor.getFromId(key).getDef():
                    reason += "graspError"
                elif candidates[key][1][0] <= 1:
                    reason += "proximityError"
                else:
                    top5_keys.append(key)
                    top5_dists.append(candidates[key][1][0])
            else:
                reason += "colorError"
        candidates[key].append(reason)

    sorted_cans = sorted(zip(top5_dists, top5_keys), key=lambda x: x[1])
    top_choices = []

    for i in range(len(sorted_cans)):
        # we take the first one and last one always
        if i == 0:
            top_choices.append(sorted_cans[i])
        # Check distance to next can
        else:
            if abs(top_choices[-1][0] - sorted_cans[i][0]) <= min_dist or sorted_cans[i][0] < 1:
                candidates[sorted_cans[i][1]][2] = "proximityError"
            else:
                top_choices.append(sorted_cans[i])
        if len(top_choices) == choices:
            break
    for i in range(len(top_choices)):
        candidates[top_choices[i][1]][2] = str(i+1)

    return candidates


def positionCheck(pos, sens, limit = 0.2):
    #print("Difference is: {}".format(abs(sens[i].getValue() - math.radians(pos[i]))))
    global movementLock
    if len(pos):
        print("Target position at: {}, position is at {}".format(pos, [math.degrees(sens[i].getValue()) for i in range(len(pos))]))
        if all([abs(sens[i].getValue() - math.radians(pos[i])) < limit for i in range(len(pos))]):
            movementLock = False
            return True 
    else: 
        return False


def displayScore(display, correct, incorrect, missed, robot_correct, robot_incorrect):
    h = int(display.getHeight() / 6)
    w = int(display.getWidth() / 2)
    marginW = int(display.getWidth()*0.05)
    marginH = int(display.getHeight()*0.05)
    x = h + marginH

    display.setOpacity(1)
    display.setAlpha(1)
    display.fillRectangle(0, 0, display.getWidth(), display.getHeight())
    display.setAlpha(1)
    display.setColor(0xFF0000)
    display.setFont("Lucida Console", 64, True)
    display.drawText("Game Over!", 300, marginH)
    display.setFont("Lucida Console", 32, True)
    display.setColor(0x000000)
    display.drawText("User score:   Robot score:".format(correct), w-200, x)
    x += h
    display.drawText("Correct:         {}             {}".format(correct, robot_correct), marginW, x)
    x += h
    display.drawText("Incorrect:       {}             {}".format(incorrect, robot_incorrect), marginW, x)
    x += h
    display.fillRectangle(marginW, int(x-h/2), int(2*w-2*marginW), int(h/20))
    display.drawText("Missed: {}".format(missed), marginW, x)
    x += h
    display.setFont("Lucida Console", 48, True)
    display.drawText("Total score:     {}".format(correct-incorrect-missed+robot_correct-robot_incorrect), marginW, x)


def countCans(missed, total_cans, candidates):
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
                elif (z > 0.6 or z < 0.4) and y <= 0.88:
                    # If it's the one being grabbed then don't delete it (height is above 0.88)
                    #if candidates[can_id][2] != '1':
                    del total_cans[can_id]
            # If the can is not in the list yet, we should add it
            else:
                if y >= 0.8:
                    trueColor = can.getDef().split('_')[0].lower()
                    total_cans[can_id] = []
                    total_cans[can_id].append(trueColor)
                    # Random number or the can is already grasped
                    if random.random() <= 0.7 or y >= 0.88:
                        total_cans[can_id].append(trueColor)
                    else:
                        options = ["yellow", "red", "green"]
                        options.remove(trueColor)
                        total_cans[can_id].append(random.choice(options))
                        #total_cans.append(can.getId())
    for item in toRemove:
        item.remove()

    for keys, vals in candidates.items():
      print (keys, vals)
    print("------------------------------------")
    #print(can_num)
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
        if obj_id not in candidates.keys():
            continue
        reason = candidates[obj_id][2]
        # Check if the object is on the conveyor or not
        _, _, obj_z = supervisor.getFromId(obj_id).getField("translation").getSFVec3f()
        if obj_z > 0.6 or obj_z < 0.4 or reason == "":
            continue 
        # Assign color
        color = colors[candidates[obj_id][0][1]] if reason != "graspError" else colors[candidates[obj_id][0][0]]
        #If the error is "graspError" and the trueColor != original
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
        if reason in list(reason_dict.keys()):
            text = reason_dict[reason]
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
    global can_num, pos_choice
    can_num += 1
    #can_distances = ["000", "999", "555", "535", "515", "495", "475", "455"]
    can_distances = ["000", "999", "556", "479", "506", "490", "530"]
    can_distances.remove(pos_choice)
    can_colors = ["green", "yellow", "red"]
    pos_choice = random.choice(can_distances)
    can = "resources/" + random.choice(can_colors) + "_can_" + pos_choice + ".wbo"
    root_children.importMFNode(-1, can)
    if pos_choice not in ["000", "999"]:
        can_reposition_dict = {"556" : 0.55, "479" : 0.48, "506" : 0.51, "490" : 0.49, "530" : 0.53}
        root_children.getMFNode(-1).getField("translation").setSFVec3f([3.1, 0.87, can_reposition_dict[pos_choice]])


def endGame():
    [supervisor.step(64) for x in range(10)]
    displayScore(display_explanation, correctSort, wrongSort, missed, robot_correct, robot_incorrect)
    [supervisor.step(64) for x in range(10)]
    supervisor.simulationSetMode(0)
    

def setPoseRobot(move_dic, can_pose):
    target = []
    print(can_pose)
    try:
        movementLock = True
        poses = move_dic[can_pose]
        [motors[i].setPosition(math.radians(poses[i])) for i in range(len(poses))]
        target = poses[:]
    except KeyError:
        print("KEYERROR!")
    return target

y = 0.88
x = 3.17

# If you want the camera image
cam = True

supervisor = Supervisor()
robot = supervisor.getFromDef("UR3")
robot_connector = supervisor.getDevice("connector")
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

# Make the display transparent
# display_score = supervisor.getDevice("display_score")
# display_score.setOpacity(1)
# display_score.setAlpha(0)
# display_score.fillRectangle(0, 0, display_score.getWidth(), display_score.getHeight())

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
for i in range(len(finger_names)):  
    fingers[i] = supervisor.getDevice(finger_names[i])
    sensor_fingers[i] = supervisor.getDevice(finger_names[i] + '_sensor')
    sensor_fingers[i].enable(timestep)

distance_sensor = supervisor.getDevice("distance_sensor1") 
distance_sensor.enable(timestep)  

my_chain = ikpy.chain.Chain.from_urdf_file("resources/robot.urdf")      

target_pos = 0
targetAcquired = False
needsMovingUp = False
movementLock = False
# Busy: If it has a target
# Prepare2grap: In position for grabbing
# LiftOff: Lift it up after grabbing (so it doesn't knock over other cans)
# Sorting: Takes to appropriate bin
# Release: Releases the can
# Back2Ready
stages = {"Prepare2grap" : False, "LiftOff": False, "Sorting": False, "Release" : False, "Back2Ready": False, "GetReady" : False}
target = []

move_down_dic = {0.55 : [5.4, -124, -85, -59, 91, 90],
                 0.53 : [41.19, -120.51, -78.88, -68.88, 91.26, 145],
                 0.51 : [50, -124.12, -85.16, -59, 91.49, 145],
                 0.49 : [52.2, -120, -41, -107, 91, 145],
                 0.48 : [54.12, -121.52, -81.19, -65.69, 91, 145]}


move_up_dic = {  0.55 : [5.4, -114, -32, -110, 91, 90],
                 0.53 : [41.25, -120, -35.99, -106.99, 91, 145],
                 0.51 : [44.35, -119.60, -35.43, -106.27, 91.43, 145],
                 0.49 : [52.2, -121, -80, -35.67, 90.86, 145],
                 0.48 : [54.26, -120, -35.99, -106.99, 91, 145]}

# custom_dic = {  "ready" : [0, -90, -90, -90, 90, 90],
#                 "green" : [125, -150, -31, -76, 90, 145],
#                 "red"   : [-100, -150, -25, -76, 90, 145]}

# custom_dic = {  "ready" : [0, -90, -90, -90, 90, 90],
#                 "green" : [115, -140, -33, -90, 90, 145],
#                 "red"   : [-100, -150, -25, -76, 90, 145]}                
custom_dic = {  "ready" : [0, -90, -90, -90, 90, 90],
                "green" : [120, -125, -30, -100, 90, 145],
                "red"   : [-120, -140, -18, -100, 90, 145]}
candidates = {}

moveFingers(fingers)
setPoseRobot(custom_dic, "ready")

robot_correct, robot_incorrect = 0, 0
root_children = supervisor.getRoot().getField("children")

robot_connector = supervisor.getDevice("connector")
robot_connector.enablePresence(timestep)

while supervisor.step(timestep) != -1:

    total_cans, missed = countCans(missed, total_cans, candidates)
    candidates = pickTargets(total_cans, 3)
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
    spawn_timer += 1
    if random.randrange(0,100) % freq == 0 and spawn_timer > spawn_limit:
        if can_num < max_cans:
            generateCans()
            spawn_timer = 0
            #supervisor.getFromDef("PHYSICS").getField("mass").setSFFloat(0.1)
        #pass
    #print("Correct: {}\t Incorrect: {}\t Missed: {}\t Total: {}".format(correctSort, wrongSort, missed, correctSort-wrongSort-missed))
    #print(onConveyorRanked(total_cans))
    if cam: drawImage(camera, colors, candidates)
    if can_num >= max_cans and not bool(total_cans):
        endGame()

############################################ 
###### R O B O T     M O V E M E N T #######
############################################
    try:
        targetPrint = target_pos
    except NameError:
        targetPrint = "None"
    busy = any(stage for stage in stages.values())
    print("Stage: {} - Target: {}".format([key for key in stages.keys() if stages[key] == True], targetPrint))

    # Update variables & check if target is still available
    if busy:
        if index not in total_cans.keys() and not stages["GetReady"]:
            for key in stages.keys(): stages[key] = False
            busy = False
            setPoseRobot(custom_dic, "ready")
        else:
            can_dist = supervisor.getFromId(index).getField("translation").getSFVec3f()    

    if not busy and getFirstCan(candidates):
         index = getFirstCan(candidates) #####SETTING THE CAN, CAN BE REPALCED BY AN ACTUAL ID#####
         can_dist = supervisor.getFromId(index).getField("translation").getSFVec3f()
         target_pos = round(can_dist[2], 2)
         target = setPoseRobot(move_up_dic, target_pos) if can_dist[0] < 2 else None
         stages["Prepare2grap"] = True if target else False   

    if stages["Prepare2grap"] and positionCheck(target, sensors) and can_dist[0] < 1.5:
         stages["Prepare2grap"] = False
         target = setPoseRobot(move_down_dic, target_pos)      
         stages["LiftOff"] = True

    if stages["LiftOff"] and robot_connector.getPresence():
         stages["LiftOff"] = False
         moveFingers(fingers, "close")    
         robot_connector.lock()
         target = setPoseRobot(move_up_dic, target_pos)
         #target = setPoseRobot(custom_dic, "ready")
         stages["Sorting"] = True

    if stages["Sorting"] and positionCheck(target, sensors, 0.01):
         stages["Sorting"] = False
         # Set target to the perceived color's crate
         target = setPoseRobot(custom_dic, candidates[index][0][1])
         stages["Release"] = True

    if  stages["Release"] and positionCheck(target, sensors, 0.02):
        moveFingers(fingers, mode = "open") 
        robot_connector.unlock()
        robot_correct += 1 if candidates[index][0][1] == candidates[index][0][0] else 0
        robot_incorrect += 1 if candidates[index][0][1] != candidates[index][0][0] else 0
        stages["Release"] = False
        if not getFirstCan(candidates):
            target = setPoseRobot(custom_dic, "ready")
            stages["GetReady"] = True

    if  stages["GetReady"] and positionCheck(target, sensors, 0.01):
        stages["GetReady"] = False
