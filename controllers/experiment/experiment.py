#!/usr/bin/env python3
"""experiment controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor
import numpy as np
import cv2

import ikpy
from ikpy.chain import Chain


# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
cv2.startWindowThread()
cv2.namedWindow("preview")

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

# -145.23
   
#  -49.98
#   108
#  -153
#   -88.63
#  0.01
    
    
for i in range(len(joint_names)):  
    motors[i] = robot.getDevice(joint_names[i])   
    
    sensors[i] = robot.getDevice(joint_names[i] + '_sensor')
    sensors[i].enable(timestep)
    #motors[i].setPosition(float('inf'))
    #motors[i].setVelocity(0.1)                
        
for i in range(len(finger_names)):  
    fingers[i] = robot.getDevice(finger_names[i])  


counter = 0

#camera = Camera("camera")
#camera.enable(timestep)
#camera.recognitionEnable(timestep)

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    #cameraData = camera.getImage()
    #image = np.frombuffer(cameraData, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    #print(camera.getRecognitionObjects())
    #print(cameraData)
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    
    
    ##############INVERSE KINEMATICS TO TEST################################
    
    
    #target_position = [position of the can + offset]
    
    
    #joints = my_chain.inverse_kinematics(target_position))
    
    #for i in range(len(self.joint_names)):
    #    self.motors[i].setPosition(joints[i+1])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print(round(sensors[0].getValue(), 1), round(sensors[1].getValue(),1), round(sensors[2].getValue(),1), round(sensors[3].getValue(),1), round(sensors[4].getValue(),1), round(sensors[5].getValue(),1))    

    counter = counter + 1
    
    motors[0].setPosition(3.14)
    motors[1].setPosition(-1.1)
    motors[2].setPosition(1.8849555922)
    motors[3].setPosition(-2.3)
    motors[4].setPosition(-1.546885316)
    motors[5].setPosition(0.01)
    
    motors[0].setVelocity(1)
    motors[1].setVelocity(0.2)   
    
    fingers[0].setPosition(0.02)
    fingers[1].setPosition(0.02)
    
    if round(sensors[0].getValue(), 1) == 3.1 and round(sensors[1].getValue(), 1) == -1.1 and round(sensors[2].getValue(),1) == 1.9 and round(sensors[3].getValue(),1) == -2.3 and round(sensors[4].getValue(),1) == -1.5 and round(sensors[5].getValue(),1) == 0.0:
        fingers[0].setPosition(0)
        fingers[1].setPosition(0)
    else:
        fingers[0].setPosition(0.02)
        fingers[1].setPosition(0.02)    
    
    
    if counter > 150:
        motors[1].setPosition(-1.5)
        
    if counter > 300:
        motors[0].setPosition(4.61) 
    if counter > 400:
        fingers[0].setPosition(0.02)
        fingers[1].setPosition(0.02)
       
        
    
    
    # Process sensor data here.
   # cv2.imshow("preview", image)
    #cv2.waitKey(timestep)
    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
