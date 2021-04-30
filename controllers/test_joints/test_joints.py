#!/usr/bin/env python3.8

from controller import Robot, Motor, Supervisor
import math

def deg2rad(deg):
    return deg*(math.pi/180)

def set_joints(pos, motors, joint_names):
    target = {'shoulder_pan_joint': pos[0],
            'shoulder_lift_joint' : pos[1],
            'elbow_joint'         : pos[2],
            'wrist_1_joint'       : pos[3],
            'wrist_2_joint'       : pos[4],
            'wrist_3_joint'       : pos[5]}

    for i in range(len(motors)):
        motors[i].setPosition(deg2rad(target[joint_names[i]]))
    while True:
        supervisor.step(TIME_STEP)
        if positionCheck(pos, sensors): break



def positionCheck(pos, sens, limit = 0.2):
    goodness = [abs(sens[i].getValue() - deg2rad(pos[i])) < limit for i in range(len(pos))]
    return all(goodness)


def moveFingers(fingers, mode="open"):
    
    if mode == "open":
        pos = [0.04, 0.04]
    elif mode == "close":
        pos = [0, 0]
    fingers[0].setPosition(pos[0])
    fingers[1].setPosition(pos[1])
    while True:
        supervisor.step(TIME_STEP)
        if positionCheck(pos, sensor_fingers, limit = 0.03): break

TIME_STEP = 16

supervisor = Supervisor()
robot_node = supervisor.getFromDef("UR3")

trans_field = robot_node.getField("translation")
finger_names = ['right_finger', 'left_finger']  
joint_names = [ 'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint']
fingers = [0] * len(finger_names)
sensor_fingers = [0] * len(finger_names)
           
motors = [0] * len(joint_names)
sensors = [0] * len(joint_names)

for i in range(len(finger_names)):  
    fingers[i] = supervisor.getDevice(finger_names[i])
    sensor_fingers[i] = supervisor.getDevice(finger_names[i] + '_sensor')
    sensor_fingers[i].enable(TIME_STEP)

for i in range(len(joint_names)):
    motors[i] = supervisor.getDevice(joint_names[i])
    motors[i].setVelocity(1)
    sensors[i] = supervisor.getDevice(joint_names[i]+'_sensor')
    sensors[i].enable(TIME_STEP)



pos555_down = [5.4, -124, -85, -59, 91, 90]
pos555_up = [5.4, -120, -41, -107, 91, 90]

pos530_down = [41.19, -120.51, -78.88, -68.88, 91.26, 90] 
pos530_up = [41.25296124941927, -120.00257244338015, -40.99851069260167, -106.99872232509098, 91.000021811653, 90]


custom_down = [44.36, -124.12, -85.16, -59, 91.49, 90]
custom_up = [44.35, -119.60, -42.43, -106.27, 91.43, 90]


custom_down2 = [54.12, -121.52, -81.19, -65.69, 91.09, 90]
custom_up2 = [54.26511925573783, -120.00257244338015, -40.99851069260167, -106.99872232509098, 91.000021811653, 90]

pos490_up = [52.20, -120, -41, -107, 91, 90]
pos490_down = [52.20, -121, -80, -65.67, 90.86, 90]

pos_up = pos490_up
pos_down = pos490_down

moveFingers(fingers, mode="open")
#[supervisor.step(TIME_STEP) for x in range(10)]
set_joints(pos_up, motors, joint_names)
print("Up reached")
#[supervisor.step(TIME_STEP) for x in range(100)]  
set_joints(pos_down, motors, joint_names)
#[supervisor.step(TIME_STEP) for x in range(10)]
#supervisor.simulationSetMode(0) 
moveFingers(fingers, mode="close")
#[supervisor.step(TIME_STEP) for x in range(10)] 
set_joints(pos_up, motors, joint_names)
#[supervisor.step(TIME_STEP) for x in range(10)] 




print(math.degrees(0.91))
print(math.degrees(-2.09444))
print(math.degrees(-0.715559))
print(math.degrees(-1.86748))
print(math.degrees(1.58825))