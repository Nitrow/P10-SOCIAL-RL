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
max_cans = 20  # 20 is doable with 50 freq
freq = 50  # Less is more - 50 is doable

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
    return None


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
        if candidates[key][1][0] > 0.5:
            if val[1] in ["green", "red"]:
                if abs(supervisor.getFromId(key).getField("rotation").getSFRotation()[3]) > 0.75:
                    reason += "graspError"
                else:
                    top5_keys.append(key)
                    top5_dists.append(candidates[key][1][0])
            else:
                reason += "colorError"

        candidates[key].append(reason)
    #top_choices = sorted(zip(top5_dists, top5_keys), key=lambda x: x[1])[:choices]

    sorted_cans = sorted(zip(top5_dists, top5_keys), key=lambda x: x[1])
    
    top_choices = []

    for i in range(len(sorted_cans)):
        # we take the first one and last one always
        if i == 0:
            top_choices.append(sorted_cans[i])
        # Check distance to next can
        else:
            if abs(top_choices[-1][0] - sorted_cans[i][0]) <= min_dist:
                candidates[sorted_cans[i][1]][2] = "proximityError"
            else:
                top_choices.append(sorted_cans[i])
        if len(top_choices) == choices:
            break
    for i in range(len(top_choices)):
        candidates[top_choices[i][1]][2] = str(i+1)

    return candidates


def position_Checker():
    """
    Returns true if all joints are 0.02 rad within the desired angles
    """
    return all([abs(sensors[i].getValue() - joints[i+1]) < 0.02 for i in range(len(sensors))])


def displayScore(display, correct, incorrect, missed):
    h = int(display.getHeight() / 6)
    w = int(display.getWidth() / 2)
    marginW = int(display.getWidth()*0.05)
    marginH = int(display.getHeight()*0.05)
    x = h + marginH
    robot_correct, robot_incorrect = 0, 0

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
    display.drawText("Total score:     {}".format(correct-incorrect-missed), marginW, x)


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
                elif z > 0.6 or z < 0.4:
                    # If it's the one being grabbed then don't delete it
                    if candidates[can_id][2] != '1':
                        del total_cans[can_id]
            # If the can is not in the list yet, we should add it
            else:
                if y >= 0.8:
                    trueColor = can.getDef().split('_')[0].lower()
                    total_cans[can_id] = []
                    total_cans[can_id].append(trueColor)
                    if random.random() <= 0.7:
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


def endGame():
    [supervisor.step(64) for x in range(10)]
    displayScore(display_explanation, correctSort, wrongSort, missed)
    supervisor.step(64)
    supervisor.simulationSetMode(0)
    

def setPoseRobot(candidates, move_dic, index):
    position_of_can = round(supervisor.getFromId(index).getField("translation").getSFVec3f()[2], 2)
    try:
        poses = move_down_dic[position_of_can]
        [motors[i].setPosition(math.radians(poses[i])) for i in range(len(poses))]
    except KeyError:
        return


def setPoseRobotUP():      
    for key, val in candidates.items():
        if val[2] == '1':
            position_of_can = val[1]
            if round(position_of_can[2],2) == 0.56:
                    for i in range(6):
                        motors[i].setPosition(math.radians(posesUP[0][i]))
             
            elif round(position_of_can[2],2) == 0.53:
                    for i in range(6):
                        motors[i].setPosition(math.radians(posesUP[1][i]))         
                         
            elif round(position_of_can[2],2) == 0.51:
                    for i in range(6):
                        motors[i].setPosition(math.radians(posesUP[2][i]))
                     
            elif round(position_of_can[2],2) == 0.49:
                    for i in range(6):
                        motors[i].setPosition(math.radians(posesUP[3][i]))  
                                         
            elif round(position_of_can[2],2) == 0.48:
                    for i in range(6):
                        motors[i].setPosition(math.radians(posesUP[4][i])) 
      
            
def setPoseRobotDOWN():

            if round(target[2],2) == 0.56:
                    for i in range(6):
                        motors[i].setPosition(math.radians(posesDOWN[0][i]))
                        
            elif round(target[2],2) == 0.53:
                    for i in range(6):
                        motors[i].setPosition(math.radians(posesDOWN[1][i]))
                        
            elif round(target[2],2) == 0.51:
                    for i in range(6):
                        motors[i].setPosition(math.radians(posesDOWN[2][i]))
                         
            elif round(target[2],2) == 0.49:
                    for i in range(6):
                        motors[i].setPosition(math.radians(posesDOWN[3][i]))  
                     
            elif round(target[2],2) == 0.48:
                    for i in range(6):
                        motors[i].setPosition(math.radians(posesDOWN[4][i]))
    



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
motors[0].setVelocity(3.14)         
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

move_down_dic = {0.57 : [5.4, -124, -85, -59, 91, 90],
                 0.53 : [41.19, -120.51, -78.88, -68.88, 91.26, 90],
                 0.51 : [44.36, -124.12, -85.16, -59, 91.49, 90],
                 0.49 : [52.2, -120, -41, -107, 91, 90],
                 0.48 : [54.12, -121.52, -81.19, -65.69, 91, 90]}

posesUP =   [[5.4, -120, -41, -107, 91, 90],
            [41.252, -120, -41, -107, 91, 90],
            [44.35, -119.60, -42.43, -106.27, 91.43, 90],
            [52.20, -120, -41, -107, 91, 90], 
            [54.2,-120, -41, -107, 91, 90]]

move_up_dic = {  0.57 : [5.4, -120, -41, -107, 91, 90],
                 0.53 : [41.25, -120, -40.99, -106.99, 91, 90],
                 0.51 : [44.35, -119.60, -42.43, -106.27, 91.43, 90],
                 0.49 : [52.2, -121, -80, -65.67, 90.86, 90],
                 0.48 : [54.26, -120, 40.99, -106.99, 91, 90]}

posesDOWN = [[5.4, -124, -85, -59, 91, 90],
            [41.19, -120.51, -78.88, -68.88, 91.26, 90],
            [44.36, -124.12, -85.16, -59, 91.49, 90],
            [52.20, -121, -80, -65.67, 90.86, 90],
            [54.12, -121.52, -81.19, -65.69, 91.09, 90]]

candidates = {}

moveFingers(fingers) 


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
    print(robot_connector.getPresence())
    # for keys, vals in total_cans.items():
    #     print (keys, vals)
    if "CAN" in selectionName:
        canSelection = selection
        canColor = selectionColor
        
    elif ("CRATE" in selectionName) and canSelection:
        new_position = selection.getField("translation").getSFVec3f()
        new_position[1] = can_height
        canSelection.getField("translation").setSFVec3f(new_position)
        changeMass(canSelection, 3)
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
####################################################################################################################### 
#######################################################################################################################
#######################################################################################################################

    if prepare_grasp == True and getFirstCan(candidates):
         
         index = getFirstCan(candidates) #####SETTING THE CAN, CAN BE REPALCED BY AN ACTUAL ID#####
         goal = supervisor.getFromId(index).getField("translation")
         target = np.array(goal.getSFVec3f())
         
         setPoseRobot(candidates, move_up_dic, index)     
        
         setPoseRobotUP()
         prepare_grasp = False
         

    if  prepare_grasp == False:
                 
         if round(target[2],2) == 0.56 and round(target[0], 2) == -0.07 + 1.02:
                      
             setPoseRobotDOWN()
             prepare_grap2 = True  
         if round(target[2],2) == 0.53 and round(target[0], 2) == 0.22 + 1.02:
                      
             setPoseRobotDOWN()
             prepare_grap2 = True
             
             
         if round(target[2],2) == 0.51 and round(target[0], 2) == 0.25 + 1.02:
                      
             setPoseRobotDOWN()
             prepare_grap2 = True
         if round(target[2],2) == 0.49 and round(target[0], 2) == 0.27 + 1.02:
             
         
             setPoseRobotDOWN()
             prepare_grap2 = True
         if round(target[2],2) == 0.48 and round(target[0], 2) == 0.31 + 1.02:
                      
             setPoseRobotDOWN()
             prepare_grap2 = True              
         setPoseRobot(candidates, move_down_dic, index)
         prepare_grap2 = True        

    if  prepare_grap2 == True and robot_connector.getPresence():
         motors[1].setPosition(sensors[1].getValue())

         moveFingers(fingers, "close")    
         robot_connector.lock() 

         go_to_bucket = True        
         prepare_grap2 = False

    if  go_to_bucket == True and go_to_bucket2 == False:

         setPoseRobotUP()
         
         
    if go_to_bucket == True and go_to_bucket2 == False and sensors[1].getValue() > -2.1 :
                 go_to_bucket2 = True    
                 go_to_bucket = False
    
    
    
    if go_to_bucket2 == True:
          go_to_bucket2 = False  
          
          if total_cans[index][1] == "green":
              for i in range(len(joint_names)):
                      motors[0].setPosition(2.2)
                      drop = True

          elif total_cans[index][1] == "red":
              for i in range(len(joint_names)):
                      motors[0].setPosition(-2)
                      drop = True

       
    if  drop == True and sensors[0].getValue()-0.01 < 2.2 < sensors[0].getValue()+0.01 or sensors[0].getValue()-0.01 < -2 < sensors[0].getValue()+0.01:
        moveFingers(fingers, mode = "open") 
        robot_connector.unlock()

        for x in range(5):
            supervisor.step(timestep)
        go_to_bucket2 == False
        prepare_grasp = True
        drop = False
    if getFirstCan(candidates):
    
        
        index = getFirstCan(candidates)
        goal = supervisor.getFromId(index).getField("translation")
        target = np.array(goal.getSFVec3f())
