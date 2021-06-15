#!/usr/bin/env python3.8

from controller import Robot, Motor, Supervisor


import pyikfastur3




import torch
from ikpy.chain import Chain
import matplotlib.pyplot as plt
import random
import sys




IKPY_MAX_ITERATIONS = 4


TIME_STEP = 32
MAX_SPEED = 6.28

counter = 0
max_iter = 10
0

supervisor = Supervisor()
robot_node = supervisor.getFromDef("UR3")



#trans_field = robot_node.getField("translation")
counter = 1

joint_names = [ 'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint']
                
motors = [0] * len(joint_names)
sensors = [0] * len(joint_names)
 
for i in range(len(joint_names)):
    # Get motors
    motors[i] = supervisor.getDevice(joint_names[i])
    motors[i].setPosition(float('inf'))
    #motors[i].setPosition(float('inf'))
    #motors[i].setVelocity(1 * MAX_SPEED)

    # Get sensors and enable them
    sensors[i] = supervisor.getDevice(joint_names[i]+'_sensor')
    sensors[i].enable(TIME_STEP)



#ikResults = armChain.inverse_kinematics([0.5, 0.5, 0.5], max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)
dir = 2

t = supervisor.getTime()
    

while supervisor.step(TIME_STEP) != -1:
    counter += 1
    t = supervisor.getTime()
    #print(counter)

    target_translation = [0.5, 0.5, 0.5]
    target_rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    # Calculate inverse kinematics
    positions = pyikfast.inverse(target_translation, target_rotation)
    print(positions)    
    
    
    for i in range(6):
        motors[i].setPosition(ikResults[i+1])
    
    
    #print(robot_node.getNumberOfContactPoints(True))
        
    if counter > max_iter:
        counter = 0
        dir *=-1
        
        #supervisor.simulationReset()
        #conveyor_node.restartController()
        #tv_node.restartController()
    # read sensors outputs
    # print("Sensor outputs:\n")
    # readings = ""
    # for i in range(len(joint_names)):
        # readings += '\t' + str(sensors[i].getValue())
    # print(readings + '\n')
