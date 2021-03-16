#!/usr/bin/env python3
"""


 WeBots tutorial: 
 https://cyberbotics.com/doc/guide/ure
"""


from controller import Robot, Motor, Supervisor
import sys
#import ikpy

# create the Robot instance.
#robot = Robot()

TIME_STEP = 32
MAX_SPEED = 6.28

counter = 0
max_iter = 100

supervisor = Supervisor()
robot_node = supervisor.getFromDef("UR3")
conveyor_node = supervisor.getFromDef("conveyor")
tv_node = supervisor.getFromDef("TV")

if robot_node is None:
    sys.stderr.write("No DEF MY_ROBOT node found in the current world file\n")
    sys.exit(1)
    
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
    motors[i].setPosition(0)
    #motors[i].setPosition(float('inf'))
    #motors[i].setVelocity(1 * MAX_SPEED)
    
    # Get sensors and enable them
    sensors[i] = supervisor.getDevice(joint_names[i]+'_sensor')
    sensors[i].enable(TIME_STEP)


while supervisor.step(TIME_STEP) != -1:
    counter += 1
    print(counter)
    if counter > max_iter:
        counter = 0
        supervisor.simulationReset()
        conveyor_node.restartController()
        tv_node.restartController()
    # read sensors outputs
    # print("Sensor outputs:\n")
    # readings = ""
    # for i in range(len(joint_names)):
        # readings += '\t' + str(sensors[i].getValue())
    # print(readings + '\n')
