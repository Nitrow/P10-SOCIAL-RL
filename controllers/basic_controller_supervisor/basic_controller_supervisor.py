#!/usr/bin/env python3.8
"""


 WeBots tutorial: 
 https://cyberbotics.com/doc/guide/ure
"""


from controller import Robot, Motor, Supervisor

import ikpy
import torch
from ikpy.chain import Chain
import matplotlib.pyplot as plt
import ikpy.utils.plot as plot_utils

# create the Robot instance.
#robot = Robot()

my_chain = Chain.from_urdf_file("../../resources/robot.urdf")
target_orientation = [0, 0, 1]
target_position = [ 0.1, -0.2, 0.1]
#target_joints = my_chain.inverse_kinematics(target_position)
target_joints = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="X")

def set_joints(target, motors):
    
    target_joints = my_chain.inverse_kinematics(target)
    #print("Moving to position {}\t{}".format(target, target_joints))
    for i in range(len(motors)):
        motors[i].setPosition(target_joints[i+1])


TIME_STEP = 32
MAX_SPEED = 6.28

counter = 0
max_iter = 100

supervisor = Supervisor()
robot_node = supervisor.getFromDef("UR3")
conveyor_node = supervisor.getFromDef("conveyor")
tv_node = supervisor.getFromDef("TV")


trans_field = robot_node.getField("translation")
counter+=1

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
    motors[i].setPosition(target_joints[i])
    #motors[i].setPosition(float('inf'))
    #motors[i].setVelocity(1 * MAX_SPEED)
    
    # Get sensors and enable them
    sensors[i] = supervisor.getDevice(joint_names[i]+'_sensor')
    sensors[i].enable(TIME_STEP)

x = 0.3
y=0.3
target = [ x,x,x]


# Reset function - reset the controllers

# Reward function


"""
" Name               Play function
"
" Description: Run the step function in the environment after setting up the action
"
"""
def play(action):
    pass

while supervisor.step(TIME_STEP) != -1:
    counter += 1
    set_joints(target, motors)
    #print(counter)
    
    if (robot_node.getNumberOfContactPoints(True)):
        contactpoints = robot_node.getNumberOfContactPoints(True)
        print("{} contact points found!".format(contactpoints))
        for x in range(contactpoints):
            print('\t',robot_node.getContactPoint(x))
    
    
    y -= 0.001
    target = [x, y, x]
    # target = [ x,y,x]
    if counter > max_iter:
        counter = 0
        
        #supervisor.simulationReset()
        #conveyor_node.restartController()
        #tv_node.restartController()
    # read sensors outputs
    # print("Sensor outputs:\n")
    # readings = ""
    # for i in range(len(joint_names)):
        # readings += '\t' + str(sensors[i].getValue())
    # print(readings + '\n')
