#!/usr/bin/env python3.8
"""


 WeBots tutorial: 
 https://cyberbotics.com/doc/guide/ure
"""


from controller import Robot, Motor, Supervisor

import ikpy
import torch
from ikpy.chain import Chain
import numpy as np

# create the Robot instance.
#robot = Robot()
"""
# IKPY example
# Compute the inverse kinematics with position
ik = baxter_left_arm_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="X")

# Let's see what are the final positions and orientations of the robot
position = baxter_left_arm_chain.forward_kinematics(ik)[:3, 3]
orientation = baxter_left_arm_chain.forward_kinematics(ik)[:3, 0]

# And compare them with was what required
print("Requested position: {} vs Reached position: {}".format(target_position, position))
print("Requested orientation on the X axis: {} vs Reached orientation on the X axis: {}".format(target_orientation, orientation))

"""


ur3_chain = Chain.from_urdf_file("../../resources/robot2.urdf")
target_orientation = np.eye(3)
target_position = [ 0.1, -0.2, 0.1]
#target_joints = []
# Set the target orientation with regards to the X axis
target_joints = ur3_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="all")
# print(target_joints)

def set_joints(motors, targetPos, targetOr = np.eye(3)):
    
    target_joints = ur3_chain.inverse_kinematics(targetPos, targetOr,  orientation_mode="all")    
    # Let's see what are the final positions and orientations of the robot
    position = np.array(ur3_chain.forward_kinematics(target_joints)[:3, 3])
    orientation = np.array(ur3_chain.forward_kinematics(target_joints)[:3, 0])

    for i in range(len(motors)):
        motors[i].setPosition(target_joints[i+1])
    return position, orientation


TIME_STEP = 32
MAX_SPEED = 6.28

counter = 0
max_iter = 100

supervisor = Supervisor()
tool_node = supervisor.getFromId(856)
robot_node = supervisor.getFromDef("UR3")
conveyor_node = supervisor.getFromDef("conveyor")
tv_node = supervisor.getFromDef("TV")
tcp =supervisor.getSelected()

# print(robot_node.getPosition())
def getTCPposition():
    ur_pos = np.array(robot_node.getPosition()) # Get the origin of the robot frame
    ur_orient = np.array(robot_node.getOrientation())
    ur_orient = ur_orient.reshape(3, 3) # Reshape into a 3 by 3 rotation matrix
    ur_orient = np.transpose(ur_orient) # We need the world relative to the robot
    tcp_pos = np.array(tool_node.getPosition())
    tcp_pos = np.subtract(tcp_pos, ur_pos) # Calculate the relative translation between them
    tcp_pos = np.dot(ur_orient, tcp_pos) # Matrix multiplication to get the tcp position relative to the robot
    
    tcp_orient = np.dot(ur_orient, np.array(tool_node.getOrientation()).reshape(3,3))
    return tcp_pos, tcp_orient


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
    
    # Get sensors and enable them
    sensors[i] = supervisor.getDevice(joint_names[i]+'_sensor')
    sensors[i].enable(TIME_STEP)


targetPos = [0,0,0]
targetOr = np.array([0,0,1, 0,1,0, 1,0,0]).reshape(3,3)


while supervisor.step(TIME_STEP) != -1:
    counter += 1
    pos_tcp_kin, rot_tcp_kin = set_joints(motors, targetPos, targetOr)
    pos_tcp, rot_tcp = getTCPposition()
    #print("Position: {}\t{}".format(np.around(pos_tcp_kin, 5), np.around(pos_tcp, 5)))
    # print("Orientation: {}\n{}\n\n\n".format(np.around(rot_tcp_kin, 1), np.around(rot_tcp, 1)))

    if counter > max_iter:
        counter = 0

        