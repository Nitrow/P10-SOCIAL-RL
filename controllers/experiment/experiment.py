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
    motors[i].setPosition(float('inf'))
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

fingers[0].setPosition(0.04)
fingers[1].setPosition(0.04)


def position_Checker():
    if sensors[0].getValue()-0.02 < joints[1] < sensors[0].getValue()+0.02 and sensors[1].getValue()-0.02 < joints[2] < sensors[1].getValue()+0.02 and sensors[2].getValue()-0.02 < joints[3] < sensors[2].getValue()+0.02 and sensors[3].getValue()-0.02 < joints[4] < sensors[3].getValue()+0.02 and sensors[4].getValue()-0.02 < joints[5] < sensors[4].getValue()+0.02 and sensors[5].getValue()-0.02 < joints[6] < sensors[5].getValue()+0.02:
        return True
    else:
        return False

# Get can objects

root_children = supervisor.getRoot().getField("children")


def countCans():
    total_cans = []
    toRemove = []
    root_children_n = root_children.getCount()
    for n in range(root_children_n):
        if "CAN" in root_children.getMFNode(n).getDef():
            can = root_children.getMFNode(n)
            x, y, z = can.getField("translation").getSFVec3f()
            if not x < -1.2 and y >= 0.8:
            #root_children.getMFNode(n).remove()
                total_cans.append(can.getId())
            else:
                toRemove.append(can)
    for item in toRemove:
        item.remove()
    return total_cans


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


def drawImage(camera):
    """
    Displays the image either in a new window, or on the Display
    """
    cameraData = camera.getImage()
    image = np.frombuffer(cameraData, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    for obj in camera.getRecognitionObjects():
        size = np.array(obj.get_size_on_image()) + padding
        start_point = np.array(obj.get_position_on_image()) - (size / 2) 
        start_point =  np.array([int(n) for n in start_point])
        end_point = start_point + size
        color = np.rint(np.array(obj.get_colors())*255)
        thickness = 2
        #color = [0, 255, 0]
        color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
        image = cv2.rectangle(image, tuple(start_point), tuple(end_point), tuple(color), thickness)
    
    if cameraData:
        # Displaying the camera image directly
        #ir = display_score.imageNew(cameraData, Display.BGRA, camera.getWidth(), camera.getHeight())
        # Displaying the processed image
        cv2.imwrite('tmp.jpg', image)
        ir = display_score.imageLoad('tmp.jpg')
        display_score.imagePaste(ir, 0, 0, False)
        display_score.imageDelete(ir)
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
    total_cans = countCans()
    #print(total_cans)
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
            #total_cans.remove(canSelection.getId())
        else: wrongSort += 1 
        canSelection = None

    # Check for missed ones:
    for canID in total_cans:  
        canX, _, _ = supervisor.getFromId(canID).getField("translation").getSFVec3f()
        if canX < -1.5:
            missed += 1

    prevSelection = selection
    if random.randrange(0,100) % 20 == 0:
        generateCans()
        #pass
    #print("Correct: {}\t Incorrect: {}\t Missed: {}\t Total: {}".format(correctSort, wrongSort, missed, correctSort-wrongSort-missed))
    #print(onConveyorRanked(total_cans))
    if cam: drawImage(camera)
    
    
    if prepare_grasp == True:
        
        
        index = 6 #####SETTING THE CAN, CAN BE REPALCED BY AN ACTUAL ID#####
        
        goal = supervisor.getFromDef("GREEN_CAN1").getField("translation")
        target = np.array(goal.getSFVec3f())
                 
        target_position = [target[2]-0.19, 0.167, target[1]-0.52]
    
        orientation_axis = "Y"
        target_orientation = [0, 0, -1]
        
        joints = my_chain.inverse_kinematics(target_position, target_orientation=target_orientation, orientation_mode=orientation_axis)
        
        for i in range(len(joint_names)):
            motors[i].setPosition(joints[i+1])    
        
    
        prepare_grasp = False
        
 
            
    if  prepare_grasp == False and position_Checker()==True and distance_sensor.getValue() < 800 and target[0] < 0.19 :
         
        #target_position = [target[2]-0.2, 0.167, target[1]-0.58]
         
        #orientation_axis = "Y"
        #target_orientation = [0, 0, -1]
    
    
        #joints = my_chain.inverse_kinematics(target_position, target_orientation=target_orientation, orientation_mode=orientation_axis)
       
        for i in range(len(joint_names)):
            motors[1].setPosition(0.15)    


        
        lower_grasp = False        
        prepare_grap2 = True        
 
    
    if  prepare_grap2 == True and distance_sensor.getValue() < 200:
        
        
        
        motors[1].setPosition(sensors[1].getValue())
        fingers[0].setPosition(0)
        fingers[1].setPosition(0)
        
        
        
        go_to_bucket = True        
        prepare_grap2 = False

        
    if  go_to_bucket == True and go_to_bucket2 == False and sensor_fingers[0].getValue() < 0.005 or sensor_fingers[1].getValue() < 0.005:
        

        for i in range(len(joint_names)):
                motors[1].setPosition(-0.5)
                
        if sensors[1].getValue()-0.1 < -0.5 < sensors[1].getValue()+0.1:
                go_to_bucket2 = True    
                go_to_bucket = False
               
               
    if go_to_bucket2 == True:
         
         if index > 5:
             for i in range(len(joint_names)):
                     motors[0].setPosition(1.5)
                     if sensors[0].getValue()-0.01 < 1.5 < sensors[0].getValue()+0.01:
                         drop = True
                         go_to_bucket2 = False

         if index < 3:
             for i in range(len(joint_names)):
                     motors[0].setPosition(-1.8)
                     if sensors[0].getValue()-0.01 < -1.8 < sensors[0].getValue()+0.01:
                         drop = True
                         go_to_bucket2 = False

         
    if drop == True:
        
       fingers[0].setPosition(0.04)
       fingers[1].setPosition(0.04)
       
       if sensor_fingers[0].getValue()-0.005 < 0.03 < sensor_fingers[0].getValue()+0.005: 
           prepare_grasp = True
           drop = False
           
           
    goal = supervisor.getFromDef("GREEN_CAN1").getField("translation")
    target = np.array(goal.getSFVec3f())          
    
    
  