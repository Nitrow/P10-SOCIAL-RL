#!/usr/bin/env python3

from controller import Robot, Camera, Supervisor, Display, Connector
import numpy as np
import cv2
import random
import os
import math

# easy or hard
gameMode = "easy"
condition = "visual"
experiment_conditions = {"control" : [False, False, False],
                         "all"     : [True, True, True],
                         "visual"  : [False, False, True],
                         "written" : [True, True, False]}

# seed, maximum amount of cans, frequency, and spawn limit and conveyor speed
gameSettings = { "medium" : [10, 40, 20, 30, 0.3], "hard" : [10, 40, 3, 20, 0.4], "easy" : [2, 40, 30, 20, 0.3]}
rseed, max_cans, freq, spawn_limit, conveyor_speed = gameSettings[gameMode]  # 20 is doable with 50 freq
random.seed(rseed)
#random.seed(rseed)

robot_reward = 2

can_num = 0
spawn_timer = 0
pos_choice = "000"

drawIntention, drawRectangle, drawText = experiment_conditions[condition]
drawIntentionLimit = 1

reason_dict = { 'colorError' : "Can't sort color", 
                'graspError': "Unable to grasp", 
                'proximityError': "Can't reach in time"}


def tryGetCratePos():
    global crate_pos_img
    for obj in camera.getRecognitionObjects():
        obj_name = supervisor.getFromId(obj.get_id()).getDef()
        if obj_name == "RED_ROBOT_CRATE":
            crate_pos_img["RED_ROBOT_CRATE"] = obj.get_position_on_image()
        elif obj_name == "GREEN_ROBOT_CRATE":
            crate_pos_img["GREEN_ROBOT_CRATE"] = obj.get_position_on_image()
    #print("Trying")


def displayScore(display, correct, incorrect, missed, robot_correct, robot_incorrect):
    h = int(display.getHeight() / 6)
    w = int(display.getWidth() / 2)
    marginW = int(display.getWidth()*0.05)
    marginH = int(display.getHeight()*0.05)
    x = h + marginH

    display.setOpacity(1)
    display.setAlpha(1)
    display.fillRectangle(0, 0, display.getWidth(), display.getHeight())
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


def moveFingers(fingers, mode="open"):

    if mode == "open":
        fingers[0].setPosition(0.04)
        fingers[1].setPosition(0.04)
    elif mode == "close":
        fingers[0].setPosition(0.015)
        fingers[1].setPosition(0.015)


def getFirstCan(candidates):
    for key, val in candidates.items():
        if val[2] == '1':
            return key
    return False

def generateCans():
    global can_num, pos_choice
    can_num += 1
    can_distances = ["000", "999", "556", "479", "506", "490", "530"]
    can_distances.remove(pos_choice)
    can_colors = ["green", "yellow", "red"]
    pos_choice = random.choice(can_distances)
    can = "resources/" + random.choice(can_colors) + "_can_" + pos_choice + ".wbo"
    root_children.importMFNode(-1, can)
    if pos_choice not in ["000", "999"]:
        can_reposition_dict = {"556" : 0.55, "479" : 0.48, "506" : 0.51, "490" : 0.49, "530" : 0.53}
        root_children.getMFNode(-1).getField("translation").setSFVec3f([2.7, 0.87, can_reposition_dict[pos_choice]])


def endGame():
    [supervisor.step(64) for x in range(10)]
    displayScore(display_explanation, correctSort, wrongSort, missed, robot_correct, robot_incorrect)
    [supervisor.step(64) for x in range(10)]
    supervisor.simulationSetMode(0)


def positionCheck(pos, sens, limit = 0.1):
    global movementLock
    if len(pos):
        #print("Target position at: {}, position is at {}".format(pos, [math.degrees(sens[i].getValue()) for i in range(len(pos))]))
        if all([abs(sens[i].getValue() - math.radians(pos[i])) < limit for i in range(len(pos))]):
            movementLock = False
            return True 
    else: 
        return False


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
        perceivedColor = candidates[obj_id][0][1]
        color = colors[perceivedColor] if reason != "graspError" else colors[candidates[obj_id][0][0]]
        #If the error is "graspError" and the trueColor != original

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
            if drawRectangle:
                image = cv2.rectangle(image, tuple(start_point), tuple(end_point), tuple(color), thickness)
            if drawIntention and int(reason) in range(drawIntentionLimit+1):
                end_point = crate_pos_img["GREEN_ROBOT_CRATE"] if perceivedColor == "green" else crate_pos_img["RED_ROBOT_CRATE"]
                start_point = np.array([int(n) for n in obj.get_position_on_image()])
                image = cv2.arrowedLine(image, tuple(start_point), tuple(end_point), tuple(color), thickness)
            start_point[1] -= 20
            text = reason
        if reason in list(reason_dict.keys()):
            text = reason_dict[reason]
        if drawText:
            cv2.putText(image, text, tuple(start_point), font, 1, tuple(color), 2)

    if cameraData:
        cv2.imwrite('tmp.jpg', image)
        ir = display_explanation.imageLoad('tmp.jpg')
        display_explanation.imagePaste(ir, 0, 0, False)
        display_explanation.imageDelete(ir)


def countCansOnConveyor(missed, cansOnConveyor):
    root_children_n = root_children.getCount()
    toRemove = []  # We need to delete the cans from the simulation in the end
    for n in range(root_children_n):
        if "CAN" in root_children.getMFNode(n).getDef():
            can = root_children.getMFNode(n)
            can_id = can.getId()
            x, y, z = can.getField("translation").getSFVec3f()
            # If we already have the can, check if it should be removed
            if can_id in list(cansOnConveyor.keys()):
                if x < -0.75:# and y >= 0.8:
                    missed += 1
                    toRemove.append(can)
                    del cansOnConveyor[can_id]
                elif (z > 0.6 or z < 0.4) and y <= 0.88:
                    # If it's the one being grabbed then don't delete it (height is above 0.88)
                    del cansOnConveyor[can_id]
            # If the can is not in the list yet, we should add it
            else:
                if y >= 0.8 and (z < 0.6 and z > 0.4) and x > 1.5:
                    trueColor = can.getDef().split('_')[0].lower()
                    cansOnConveyor[can_id] = []
                    cansOnConveyor[can_id].append(trueColor)
                    # Random number or the can is already grasped
                    if random.random() <= 0.7 or y >= 0.88:
                        cansOnConveyor[can_id].append(trueColor)
                    else:
                        options = ["yellow", "red", "green"]
                        options.remove(trueColor)
                        cansOnConveyor[can_id].append(random.choice(options))
    for i in range(len(toRemove)): toRemove[i].remove()
    return cansOnConveyor, missed


def pickTargets(cansOnConveyor, choices=5, min_dist = 0.7, outOfReachLimit = 0.7):
    """
    Gets five targets based on 3 criteria (assessing each can):
    """
    candidates = {}
    top5_dists = []
    top5_keys = []
    top_choices = []

    for key, val in cansOnConveyor.items():
        reason = ""

        candidates[key] = [val]  # Add all the cans on the conveyor as candidates
        candidates[key].append([round(c,3) for c in supervisor.getFromId(key).getField("translation").getSFVec3f()])
        grasped = candidates[key][1][1] > 0.9  # If the can is elevated then it is considered grasped
        if grasped: top_choices.append([candidates[key][1][0], key])
        # The can has to be closer then 0.5 and within the range of the conveyor or be (grasped above conveyor)
        if (candidates[key][1][0] >= 0.5 and (int(candidates[key][1][2]*100) in range(40,60)) and not grasped):
            if val[1] in ["green", "red"]:
                if "ROTATED" in supervisor.getFromId(key).getDef():
                    reason += "graspError"
                elif candidates[key][1][0] <= outOfReachLimit: # + (tcp.getPosition()[0] - candidates[key][1][0]  * conveyor_speed * (timestep/1000)):
                    reason += "proximityError"
                #elif candidates[key][1][0] < 1.2 and  candidates[key][1][0] < 1 + np.linalg.norm(np.array(candidates[key][1][::2]) - np.array(tcp.getPosition()[::2])) > outOfReachLimit:
                #    reason += "proximityError"
                else:
                    top5_keys.append(key)
                    top5_dists.append(candidates[key][1][0])
            else:
                reason += "colorError"
        candidates[key].append(reason)

    sorted_cans = sorted(zip(top5_dists, top5_keys), key=lambda x: x[1])

    for i in range(len(sorted_cans)):
        # we take the first one and last one always
        
            
        if (i == 0 and candidates[key][1][0] >= 0.7):
            top_choices.append(sorted_cans[i])
        # Check distance to next can
        else:
            if abs(top_choices[-1][0] - sorted_cans[i][0]) <= min_dist or sorted_cans[i][0] < 0.6:
                candidates[sorted_cans[i][1]][2] = "proximityError"
            else:
                top_choices.append(sorted_cans[i])
        if len(top_choices) == choices:
            break
    for i in range(len(top_choices)):
        candidates[top_choices[i][1]][2] = str(i+1)



    for keys, vals in candidates.items():
        print (keys, vals)
    print("------------------------------------")

    return candidates
    

def setPoseRobot(move_dic, can_pose):
    target = []

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

supervisor = Supervisor()
robot = supervisor.getFromDef("UR3")
robot_connector = supervisor.getDevice("connector")

timestep = int(supervisor.getBasicTimeStep())
tcp = supervisor.getFromDef('TCP')
cr_s = conveyor_speed * 3
supervisor.getFromDef('CONVEYOR').getField("speed").setSFFloat(conveyor_speed)  # Get the speed of the conveyor to scale the distances
padding = np.array([10, 10])
camera = Camera("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)

display_explanation = supervisor.getDevice("display_explanation")


width = camera.getWidth()
height = camera.getHeight()

colors = {"yellow" : [0.309804, 0.913725, 1.0], "red" : [0.0, 0.0, 1.0], "green" : [0.0, 1.0, 0.0]}

selection = None
prevSelection = None
canSelection = None
can_height = 0.85
selectionName = None
canSelectionName = None

cansOnConveyor = {}

missed = 0
correctSort = 0
wrongSort = 0

# Crate position on the recognition image
crate_pos_img = {"RED_ROBOT_CRATE" : [], "GREEN_ROBOT_CRATE" : []}

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

target_pos = 0
targetAcquired = False
needsMovingUp = False
movementLock = False

stages = {"Prepare2grap" : False, "LiftOff": False, "Sorting": False, "Release" : False, "Back2Ready": False, "GetReady" : False}
target = []


move_down_dic_l = {0.55 : [-12, -124.8, -67.83, -77.53, 92.25, 84],
                 0.53 : [-21.88, -132.16, -65.76, -70.54, 91.87, 64.25],
                 0.51 : [-22.86, -128.28, -72.68, -67.54, 91.88, 64.25 ],
                 0.49 : [-23.88, -124.65, -78.73, -65.13, 91.9, 64.25],
                 0.48 : [-24.48, -122.99, -81.99, -63.54, 91.91, 64.25]}

move_up_dic_l = {  0.55 : [-12, -132.15, -17.5, -120.5, 92.17, 83.78],
                 0.53 : [-21.89, -129.37, -44.87, -94.22, 91.83, 64.25],
                 0.51 : [-22.85, -124.38, -51.91, -92.20, 91.85, 64.25],
                 0.49 : [-23.88, -120.9, -48.42, -99.2, 91.85, 64.25],
                 0.48 : [-24.47, -118.4, -52.34, -97.79, 91.87, 64.25]}


move_down_dic_r = {0.55 : [44.11, -128.46, -62.3, -77.9, 91, 140],
                 0.53 : [46, -124, -70, -75, 91, 140],
                 0.51 : [48, -120, -76.5, -72, 91, 140],
                 0.49 : [53.3, -120, -72.4, -76, 91, 140],
                 0.48 : [54.12, -121.52, -81.19, -65.69, 91, 140]}


move_up_dic_r = {  0.55 : [44.1, -130.2, -33, -105.4, 91.2, 140],
                 0.53 : [46, -125.26, -37, -106.43, 91, 140],
                 0.51 : [48, -121.5, -37, -110, 91, 140],
                 0.49 : [53.3, -123, -35.5, -110, 90, 140],
                 0.48 : [54.11, -119, -46.48, -102.99, 91, 140]}

   
custom_dic = {  "ready" : [0, -90, -90, -90, 90, 90],
                "green" : [120, -125, -30, -115, 90, 145],
                "red"   : [-107.77, -119, -46.48, -102.99, 91, 145]}

candidates = {}

move_dict = {"up" : move_up_dic_r, "down" : move_down_dic_r}

moveFingers(fingers)
setPoseRobot(custom_dic, "ready")

graspedCan = None

robot_correct, robot_incorrect = 0, 0
root_children = supervisor.getRoot().getField("children")

robot_connector = supervisor.getDevice("connector")
robot_connector.enablePresence(timestep)

can_tcp_dist = 999


while supervisor.step(timestep) != -1:
    if not bool(crate_pos_img["GREEN_ROBOT_CRATE"]): tryGetCratePos()
    cansOnConveyor, missed = countCansOnConveyor(missed, cansOnConveyor)

    candidates = pickTargets(cansOnConveyor, 3)

    selection = supervisor.getSelected()
    selectionName = selection.getDef() if selection else ""
    selectionColor = selectionName.split('_')[0]

    if "CAN" in selectionName:
        canSelection = selection
        canColor = selectionColor
        
    elif ("CRATE" in selectionName) and canSelection:
        if canSelection.getId() == graspedCan and robot_connector.isLocked():
            canSelection = None
            continue
        new_position = selection.getField("translation").getSFVec3f()
        new_position[1] = can_height
        canSelection.getField("translation").setSFVec3f(new_position)
        if selectionColor == canColor: correctSort += 1
        else: wrongSort += 1 
        del cansOnConveyor[canSelection.getId()]
        canSelection = None

    # Check for missed ones:
    for canID in cansOnConveyor:  
        canX, _, _ = supervisor.getFromId(canID).getField("translation").getSFVec3f()
        if canX < -1.5:
            missed += 1

    prevSelection = selection
    spawn_timer += 1
    if random.randrange(0,100) % freq == 0 and spawn_timer > spawn_limit:
        if can_num < max_cans:
            generateCans()
            spawn_timer = 0

    drawImage(camera, colors, candidates)
    if can_num >= max_cans and not bool(cansOnConveyor):
        endGame()
############################################ 
###### R O B O T     M O V E M E N T #######
############################################
    try:
        targetPrint = target_pos
        targetIndex = index
    except NameError:
        targetPrint = "None"
        targetIndex = "None"
    busy = any(stage for stage in stages.values())
    stage = [key for key in stages.keys() if stages[key] == True]
    print("Stage: {} - Target: {} Index {}".format([key for key in stages.keys() if stages[key] == True], targetPrint, targetIndex))

    # Update variables & check if target is still available
    if busy:
        if stage not in ["Sorting", "Release", "LiftOff"] and not robot_connector.isLocked():
            move_dict = {"up" : move_up_dic_r, "down" : move_down_dic_r} if (can_dist[0] > (max(0.8*cr_s,1.22)) and can_tcp_dist >= 0) else {"up" : move_up_dic_l, "down" : move_down_dic_l} 
            if index not in cansOnConveyor.keys() or candidates[index][2] != "1":
                print("++++++++++++++ INTERRUPTED ++++++++++++++")
                for key in stages.keys(): stages[key] = False
                busy = False
                if not getFirstCan(candidates): target = setPoseRobot(custom_dic, "ready")
            else:
                can_dist = supervisor.getFromId(index).getField("translation").getSFVec3f()
                can_tcp_dist = can_dist[0]-tcp.getPosition()[0]

    if not busy and getFirstCan(candidates):
         index = getFirstCan(candidates) #####SETTING THE CAN, CAN BE REPALCED BY AN ACTUAL ID#####
         can_dist = supervisor.getFromId(index).getField("translation").getSFVec3f()
         can_tcp_dist = can_dist[0]-tcp.getPosition()[0]
         target_pos = round(can_dist[2], 2)
         target = setPoseRobot(move_dict["up"], target_pos) if abs(can_tcp_dist) < 0.75*cr_s else None
         stages["GetReady"] = True if target else False   

    if stages["GetReady"] and positionCheck(target, sensors):
         target = setPoseRobot(move_dict["up"], target_pos)
         print(can_tcp_dist)

    if stages["GetReady"] and positionCheck(target, sensors) and 0 < (can_tcp_dist) < 0.1*cr_s :
         
         stages["GetReady"] = False
         stages["Prepare2grap"] = True
         target = setPoseRobot(move_dict["down"], target_pos)      
    
    if stages["Prepare2grap"] and (can_tcp_dist) < -0.05:
         stages["Prepare2grap"] = False
         stages["GetReady"] = True

    if stages["Prepare2grap"] and robot_connector.getPresence():# and (can_dist[0]-tcp.getPosition()[0]) > 0.05:
         stages["Prepare2grap"] = False
         moveFingers(fingers, "close")    
         robot_connector.lock()
         graspedCan = index
         target = setPoseRobot(move_dict["up"], target_pos)
         #target = setPoseRobot(custom_dic, "ready")
         stages["LiftOff"] = True

    if stages["LiftOff"] and positionCheck(target, sensors, 0.01):
         stages["LiftOff"] = False
         # Set target to the perceived color's crate
         stages["Sorting"] = True
         target = setPoseRobot(custom_dic, candidates[index][0][1])

    if  stages["Sorting"] and positionCheck(target, sensors, 0.01) and robot_connector.isLocked():
        stages["Sorting"] = False
        stages["Release"] = True
        moveFingers(fingers, mode = "open") 
        robot_connector.unlock()
        target = setPoseRobot(custom_dic, candidates[index][0][1])
        robot_correct += robot_reward if candidates[index][0][1] == candidates[index][0][0] else 0
        robot_incorrect += robot_reward if candidates[index][0][1] != candidates[index][0][0] else 0
        

    if  stages["Release"] and positionCheck(target, sensors, 0.01) and not robot_connector.isLocked() :
        target = setPoseRobot(custom_dic, "ready")
        stages["Release"] = False
        stages["GetReady"] = True
