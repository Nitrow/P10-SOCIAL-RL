#!/usr/bin/env python3
"""experiment controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor, Display, MouseState, Mouse
import numpy as np

import ikpy
from ikpy.chain import Chain

supervisor = Supervisor()
robot = supervisor.getFromDef("UR3")

timestep = int(supervisor.getBasicTimeStep())


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


list_of_cans = ['RED_CAN1', 'RED_CAN2', 'RED_CAN3', 'YELLOW_CAN1', 'YELLOW_CAN2', 'YELLOW_CAN3', 'GREEN_CAN1', 'GREEN_CAN2', 'GREEN_CAN3']

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
    
my_chain = ikpy.chain.Chain.from_urdf_file("/home/asger/P10-XRL/resources/robot2.urdf")


prevClick = False
doubleClick = False
selection = None
prevSelection = None
canSelection = None
can_height = 0.85
prepare_grasp = True
go_to_bucket = False
prepare_grap2 = False
go_to_bucket2 = False
drop = False

fingers[0].setPosition(0.05)
fingers[1].setPosition(0.05)


def position_Checker():
    if sensors[0].getValue()-0.02 < joints[1] < sensors[0].getValue()+0.02 and sensors[1].getValue()-0.02 < joints[2] < sensors[1].getValue()+0.02 and sensors[2].getValue()-0.02 < joints[3] < sensors[2].getValue()+0.02 and sensors[3].getValue()-0.02 < joints[4] < sensors[3].getValue()+0.02 and sensors[4].getValue()-0.02 < joints[5] < sensors[4].getValue()+0.02 and sensors[5].getValue()-0.02 < joints[6] < sensors[5].getValue()+0.02:
        return True
    else:
        return False
    
while supervisor.step(timestep) != -1:
        
    
    

    if prepare_grasp == True:
        
        
        index = 6 #####SETTING THE CAN, CAN BE REPALCED BY AN ACTUAL ID#####
        
        goal = supervisor.getFromDef(list_of_cans[index]).getField("translation")
        target = np.array(goal.getSFVec3f())
                 
        target_position = [target[2]-0.21, 0.167, target[1]-0.52]
    
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
 
    
    if  prepare_grap2 == True and distance_sensor.getValue() < 300:
        
        
        
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
        
       fingers[0].setPosition(0.05)
       fingers[1].setPosition(0.05)
       
       if sensor_fingers[0].getValue()-0.005 < 0.03 < sensor_fingers[0].getValue()+0.005: 
           prepare_grasp = True
           drop = False
           
           
    goal = supervisor.getFromDef(list_of_cans[index]).getField("translation")
    target = np.array(goal.getSFVec3f())           
